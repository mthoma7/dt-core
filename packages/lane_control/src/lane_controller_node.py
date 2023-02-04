#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import (
    Twist2DStamped,
    LanePose,
    WheelsCmdStamped,
    BoolStamped,
    FSMState,
    StopLineReading,
    AprilTagsWithInfos,
)

#***********************************************************************************
import requests
#***********************************************************************************

from lane_controller.controller import LaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocities, by processing the estimate error in
    lateral deviationa and heading.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:
        ~v_bar (:obj:`float`): Nominal velocity in m/s
        ~k_d (:obj:`float`): Proportional term for lateral deviation
        ~k_theta (:obj:`float`): Proportional term for heading deviation
        ~k_Id (:obj:`float`): integral term for lateral deviation
        ~k_Iphi (:obj:`float`): integral term for lateral deviation
        ~d_thres (:obj:`float`): Maximum value for lateral error
        ~theta_thres (:obj:`float`): Maximum value for heading error
        ~d_offset (:obj:`float`): Goal offset from center of the lane
        ~integral_bounds (:obj:`dict`): Bounds for integral term
        ~d_resolution (:obj:`float`): Resolution of lateral position estimate
        ~phi_resolution (:obj:`float`): Resolution of heading estimate
        ~omega_ff (:obj:`float`): Feedforward part of controller
        ~verbose (:obj:`bool`): Verbosity level (0,1,2)
        ~stop_line_slowdown (:obj:`dict`): Start and end distances for slowdown at stop lines
    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
        ~intersection_navigation_pose (:obj:`LanePose`): The lane pose estimate from intersection navigation
        ~wheels_cmd_executed (:obj:`WheelsCmdStamped`): Confirmation that the control action was executed
        ~stop_line_reading (:obj:`StopLineReading`): Distance from stopline, to reduce speed
        ~obstacle_distance_reading (:obj:`stop_line_reading`): Distancefrom obstacle virtual stopline, to reduce speed
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        # Add the node parameters to the parameters dictionary
        # TODO: MAKE TO WORK WITH NEW DTROS PARAMETERS
        self.params = dict()
        self.params["~v_bar"] = DTParam("~v_bar", param_type=ParamType.FLOAT, min_value=0.0, max_value=5.0)
        self.params["~k_d"] = DTParam("~k_d", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0)
        self.params["~k_theta"] = DTParam(
            "~k_theta", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0
        )
        self.params["~k_Id"] = DTParam("~k_Id", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0)
        self.params["~k_Iphi"] = DTParam(
            "~k_Iphi", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0
        )
        self.params["~theta_thres"] = rospy.get_param("~theta_thres", None)
        self.params["~d_thres"] = rospy.get_param("~d_thres", None)
        self.params["~d_offset"] = rospy.get_param("~d_offset", None)
        self.params["~integral_bounds"] = rospy.get_param("~integral_bounds", None)
        self.params["~d_resolution"] = rospy.get_param("~d_resolution", None)
        self.params["~phi_resolution"] = rospy.get_param("~phi_resolution", None)
        self.params["~omega_ff"] = rospy.get_param("~omega_ff", None)
        self.params["~verbose"] = rospy.get_param("~verbose", None)
        self.params["~stop_line_slowdown"] = rospy.get_param("~stop_line_slowdown", None)

        # Need to create controller object before updating parameters, otherwise it will fail
        self.controller = LaneController(self.params)
        # self.updateParameters() # TODO: This needs be replaced by the new DTROS callback when it is implemented

        # Initialize variables
        self.fsm_state = None
        self.wheels_cmd_executed = WheelsCmdStamped()
        self.pose_msg = LanePose()
        self.pose_initialized = False
        self.pose_msg_dict = dict()
        self.last_s = None
        self.stop_line_distance = None

        # ADDED Variables for CHARGING
        self.charge_stop = False
        self.tag_pose_msg = None
        self.lane_pose_msg = None
        self.min_charge = 98
        self.tag_info = False

        self.current_pose_source = "lane_filter"

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher(
            "~car_cmd", Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL
        )

        # ADDED PUBLISHER FOR CHARGING
        self.pub_bat_soc = rospy.Publisher(
            "~bat_soc",
            BoolStamped,
            queue_size=1
        )

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber(
            "~lane_pose", LanePose, self.cbAllPoses, "lane_filter", queue_size=1
        )
        self.sub_intersection_navigation_pose = rospy.Subscriber(
            "~intersection_navigation_pose",
            LanePose,
            self.cbAllPoses,
            "intersection_navigation",
            queue_size=1,
        )
        self.sub_wheels_cmd_executed = rospy.Subscriber(
            "~wheels_cmd", WheelsCmdStamped, self.cbWheelsCmdExecuted, queue_size=1
        )
        
        # ADDED SUBSCRIBER FOR CHARGING
        self.sub_fsm_mode = rospy.Subscriber(
            "~fsm_mode",
            FSMState,
            self.cbFSMState,
            queue_size=1
        )

        self.sub_tag_infos = rospy.Subscriber(
            "apriltag_postprocessing_node/apriltags_out",
            AprilTagsWithInfos, self.cbTagPose
        )

        self.log("Initialized!")


    def cbFSMState(self, msg):
        if msg.state == "LANE_FOLLOWING":
            self.charge_stop = True
            self.tag_info = False
        elif msg.state == "LANE_FOLLOWING_AVOID":
            self.tag_info = True
        elif msg.state == "NORMAL_JOYSTICK_CONTROL":
            self.charge_stop = False
            self.tag_info = False


    def cbTagPose(self, tag_msg):
        self.tag_pose_msg  = None
        if self.tag_info:
            if tag_msg.detections:
                for item in tag_msg.detections:
                    if  item.tag_id == 125:
                        self.tag_pose_msg  = tag_msg.detections
                        self.getControlAction(self.lane_pose_msg, self.tag_pose_msg)



    def cbAllPoses(self, input_pose_msg, pose_source):
        """Callback receiving pose messages from multiple topics.
        If the source of the message corresponds with the current wanted pose source, it computes a control command.
        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
            pose_source (:obj:`String`): Source of the message, specified in the subscriber.
        """       
        if pose_source == self.current_pose_source:
            self.pose_msg_dict[pose_source] = input_pose_msg
            self.lane_pose_msg = input_pose_msg
            self.getControlAction(self.lane_pose_msg, self.tag_pose_msg)

    def cbWheelsCmdExecuted(self, msg_wheels_cmd):
        """Callback that reports if the requested control action was executed.
        Args:
            msg_wheels_cmd (:obj:`WheelsCmdStamped`): Executed wheel commands
        """
        self.wheels_cmd_executed = msg_wheels_cmd

    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.
        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)

        msg_bat_state = BoolStamped()
        msg_bat_state.header.stamp = rospy.Time(0)
        msg_bat_state.data = False
        
        if self.charge_stop:
            url = f"http://pumuckl1/health/battery"
            self.bat_data = requests.get(url).json()
            if self.bat_data["battery"]["percentage"] <= self.min_charge:
                msg_bat_state.data = True

        self.pub_bat_soc.publish(msg_bat_state)


    def getControlAction(self, lane_pose_msg, tag_pose_msg):
        """Callback that receives a pose message and updates the related control command.
        Using a controller object, computes the control action using the current pose estimate.
        Args:
            pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        current_s = rospy.Time.now().to_sec()
        dt = None
        if self.last_s is not None:
            dt = current_s - self.last_s

        lf = True
        lf_v_min = False

        if tag_pose_msg!=None:
            omega = 0.0

            tag_pose_trns = {"x":0,"y":0,"z":0};
            tag_pose_rot = {"x":0,"y":0,"z":0,"w":0};
            for item in tag_pose_msg:
                if item.transform:
                    tag_pose_trns["x"] = (item.transform.translation.x)
                    tag_pose_trns["y"] = (item.transform.translation.y)
                    tag_pose_trns["z"] = (item.transform.translation.z)
                    tag_pose_rot["x"] = (item.transform.rotation.x)
                    tag_pose_rot["y"] = (item.transform.rotation.y)
                    tag_pose_rot["z"] = (item.transform.rotation.z)
                    tag_pose_rot["w"] = (item.transform.rotation.w)
            
            #self.log(tag_pose_trns)
            #self.log(tag_pose_rot)

            if tag_pose_trns["x"] < 1.0:
                lf_v_min = True

            if tag_pose_trns["x"] < 0.5:
                lf = False
                lf_v_min = False
                v = 0.1
                omega_rot = -tag_pose_rot["x"]*5
                omega_trns = (0.1075+tag_pose_trns["y"])*3
                omega = omega_rot + omega_trns

            if tag_pose_trns["x"] < 0.25:
                lf = False
                lf_v_min = False
                v = 0.0
                omega = 0.0
                
            
        if lf:
            # Compute errors
            d_err = lane_pose_msg.d - self.params["~d_offset"]
            phi_err = lane_pose_msg.phi

            # We cap the error if it grows too large
            if np.abs(d_err) > self.params["~d_thres"]:
                self.log("d_err too large, thresholding it!", "error")
                d_err = np.sign(d_err) * self.params["~d_thres"]

            wheels_cmd_exec = [self.wheels_cmd_executed.vel_left, self.wheels_cmd_executed.vel_right]
            v, omega = self.controller.compute_control_action(
                d_err, phi_err, dt, wheels_cmd_exec, self.stop_line_distance
            )

            # For feedforward action (i.e. during intersection navigation)
            omega += self.params["~omega_ff"]

        if lf_v_min:
            v = 0.125

        # Initialize car control msg, add header from input message
        car_control_msg = Twist2DStamped()
        car_control_msg.header = lane_pose_msg.header

        # Add commands to car message
        car_control_msg.v = v
        car_control_msg.omega = omega

        self.publishCmd(car_control_msg)
        self.last_s = current_s

    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name="lane_controller_node")
    # Keep it spinning
    rospy.spin()