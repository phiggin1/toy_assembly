#!/usr/bin/env python3

import sys
import rospy
import numpy as np
import math
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from image_geometry import PinholeCameraModel
import sensor_msgs.point_cloud2 as pc2
from toy_assembly.srv import SAM
from toy_assembly.srv import CLIP
from toy_assembly.srv import MoveITGrabPose, MoveITPose
from toy_assembly.srv import OrientCamera

from std_srvs.srv import Trigger 

import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped 
import tf
from tf.transformations import quaternion_from_euler
import json

class Right_arm:
    def __init__(self):
        rospy.init_node("rightArm")
        #org right starting pose
        #self.start_pose = [0.0, 0.0, -1.0, 0.0, -2.0, 1.57]


        self.start_pose = [0.0, 0.0, -1.57, 0.0, -1.57, 0.0]
        self.init_position()	

        self.finger_open = 0.01
        self.finger_closed = 0.735 

        self.hand_closed = [self.finger_closed, self.finger_closed, self.finger_closed, self.finger_closed, self.finger_closed, self.finger_closed]
        self.hand_open = [self.finger_open, self.finger_open, self.finger_open, self.finger_open, self.finger_open, self.finger_open]

        self.gripper_group_name = "gripper"
        self.gripper_move_group = moveit_commander.MoveGroupCommander(self.gripper_group_name)
        self.gripper_move_group.set_max_velocity_scaling_factor(1.0)

        self.horse_topic = '/object_positions'
        self.horse_pose = None  
        #self.horses = rospy.Subscriber(self.horse_topic, PoseStamped, self.get_horse_pose)
        #self.horse_pose = []
        self.grabbed_object = False

        self.gripper_orientation = {'hand_pointing_right_cam_up': [math.sqrt(2)/2, 0, 0, math.sqrt(2)/2],
                            'hand_pointing_right_cam_front': [0.5, 0.5, -0.5, 0.5],
                            'hand_pointing_left_cam_up': [0, math.sqrt(2)/2, math.sqrt(2)/2, 0],
                            'hand_pointing_left_cam_front': [-0.5, -0.5, -0.5, -0.5],
                            'hand_pointing_down_cam_right': [-1, 0, 0, 0],
                            'hand_pointing_down_cam_front': [-math.sqrt(2)/2, -math.sqrt(2)/2, 0, 0],
                            'hand_pointing_forward_cam_up': [0.5, 0.5, 0.5, 0.5],
                            'hand_pointing_forward_cam_right' : [math.sqrt(2)/2, 0, math.sqrt(2)/2, 0]}

        self.listener = tf.TransformListener()
        self.grab = rospy.Publisher('/buttons', String, queue_size=10)
        #self.release = rospy.Publisher('/buttons', String, queue_size=10)

        self.move_pose = rospy.Service('move_pose', MoveITPose, self.move_to_pose)

        self.grab_object = rospy.Service('grab_object', MoveITGrabPose, self.get_object)
        self.release_object = rospy.Service('release_object', MoveITGrabPose, self.place_object)

        self.open_hand = rospy.Service('open_hand', Trigger, self.open_gipper)
        self.close_hand = rospy.Service('close_hand', Trigger, self.close_gipper)

        #self.change_orientation(None)

        self.rotate_object = rospy.Service('rotate_object', OrientCamera, self.change_orientation)
        
        rospy.spin()
	
    def init_position(self):
        moveit_commander.roscpp_initialize(sys.argv)
        #rospy.init_node("move_to_start", anonymous=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group_name = "arm"
        self.arm_move_group = moveit_commander.MoveGroupCommander(self.arm_group_name)
        self.planning_frame = "right_base_link"

        self.gripper_group_name = "arm"
        self.gripper_move_group = moveit_commander.MoveGroupCommander(self.gripper_group_name)

        self.arm_move_group.set_max_velocity_scaling_factor(0.750)
        #self.arm_move_group.set_goal_position_tolerance(self.goal_tolerance)
        self.arm_move_group.go(self.start_pose, wait=True) 
        self.arm_move_group.stop()

    def get_horse_pose(self, pose):
        """
        gets the 3d location of objects in space specified in the cluster
        """
        #del self.horse_pose[:]
        print(f'pose: {pose}')
        self.horse_pose = self.transform_obj_pos(pose)
        print(f'transformed: {self.horse_pose}')

    def move_to_pose(self, request):
        object_pose = self.transform_obj_pos(request.pose)        
        print(object_pose)
        pose_goal = Pose()
        #print('timeout1')
        pose_goal.position = object_pose.pose.position
        pose_goal.orientation = object_pose.pose.orientation
        self.arm_move_group.set_pose_target(pose_goal)

        #print('timeout3')
        status = False
        status = self.arm_move_group.go(pose_goal, wait = True)
        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()

        return status

    def get_object(self, request):
        object_pose = self.transform_obj_pos(request.pose)        
        #print(object_pose)
        pose_goal = Pose()
        #print('timeout1')
        pose_goal.position = object_pose.pose.position
        quat = quaternion_from_euler(math.pi, 0.0, 0.0)
        #print(quat)
        pose_goal.orientation.x = quat[0]
        pose_goal.orientation.y = quat[1]
        pose_goal.orientation.z = quat[2]
        pose_goal.orientation.w = quat[3]
        #print(pose_goal.orientation)

        #print('timeout 2')
        self.arm_move_group.set_pose_target(pose_goal)

        #print('timeout3')
        print(pose_goal)
        self.arm_move_group.go(pose_goal, wait=True)
        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()

        #print(object_pose)
        pose_goal2 = Pose()
        #print('timeout1')
        pose_goal2.position = object_pose.pose.position
        quat = quaternion_from_euler(math.pi, 0.0, 0.0)
        #print(quat)
        pose_goal2.orientation.x = quat[0]
        pose_goal2.orientation.y = quat[1]
        pose_goal2.orientation.z = quat[2]
        pose_goal2.orientation.w = quat[3]
        #print(pose_goal2.orientation)
        
        status = False
        status = self.arm_move_group.go(pose_goal2, wait = True)
        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()

        """
        publishing "grabbed" vs. "released" will do as follows in unity with the nearest object
        """

        # close fingers
        self.close_gipper()

        self.arm_move_group.set_max_velocity_scaling_factor(0.750)
        self.arm_move_group.go(self.start_pose, wait=True) 
        self.arm_move_group.stop()

        self.grabbed_object = status
        return status
        

    def close_gipper(self, req):
        """
        publishing "grabbed" vs. "released" will do as follows in unity with the nearest object
        """

        # close fingers
        a = dict()
        a["robot"] = "right"
        a["action"] = "grab"
        s = json.dumps(a)
        self.grab.publish(s)
        
        self.gripper_move_group.go(self.hand_closed, wait=True) 
        self.gripper_move_group.stop()

        
    def open_gipper(self, req):
        """
        publishing "grabbed" vs. "released" will do as follows in unity with the nearest object
        """

        # close fingers
        a = dict()
        a["robot"] = "right"
        a["action"] = "release"
        s = json.dumps(a)
        self.grab.publish(s)

        
        status = self.gripper_move_group.go(self.hand_open, wait=True) 
        self.gripper_move_group.stop()

        

    def transform_obj_pos(self, obj_pos):
        t = rospy.Time.now()
        obj_pos.header.stamp = t
        self.listener.waitForTransform(obj_pos.header.frame_id, self.planning_frame, t, rospy.Duration(4.0))
        obj_pos = self.listener.transformPose(self.planning_frame, obj_pos)
        return obj_pos

    def place_object(self, request):
        if (self.grabbed_object == True):
            human_hand = self.transform_obj_pos(request.pose)
            #print(object_pose)
            pose_goal = Pose()
            #print('timeout1')
            pose_goal.position = human_hand.pose.position
            quat = quaternion_from_euler(math.pi, 0.0, 0.0)
            #print(quat)
            pose_goal.orientation.x = quat[0]
            pose_goal.orientation.y = quat[1]
            pose_goal.orientation.z = quat[2]
            pose_goal.orientation.w = quat[3]
               
            self.arm_move_group.set_pose_target(pose_goal)

            status = False
            status = self.arm_move_group.go(pose_goal2, wait = True)
            self.arm_move_group.stop()
            self.arm_move_group.clear_pose_targets()

            # open fingers
            self.gripper_move_group.go(self.hand_open, wait=True) 
            self.gripper_move_group.stop()

            a = dict()
            a["robot"] = "right"
            a["action"] = "release"
            s = json.dumps(a)
            self.grab.publish(s)
            self.grabbed_object = status


            return status
        else:
            return False    

    def change_orientation(self, request):
        print("change_orientation")
        pose_goal = Pose()    
        pose_goal.position = self.arm_move_group.get_current_pose()
        orientationList = []

        # take request string and get corresponding orientation values
        if request.text in self.gripper_orientation.keys():
            orientationList = self.gripper_orientation.get(request.text)
        else:
            print('Invalid orientation request')
            orientationList = self.arm_move_group.get_current_pose().orientation

        # set pose goal position to current position
        current_pose = self.arm_move_group.get_current_pose().pose
        pose_goal = current_pose
                 
        # set pose goal orientation to selected orientation
        pose_goal.orientation.x = orientationList[0]
        pose_goal.orientation.y = orientationList[1]
        pose_goal.orientation.z = orientationList[2]
        pose_goal.orientation.w = orientationList[3]
            
        self.arm_move_group.set_pose_target(pose_goal)

        # move the arm according to the orientation goal and 
        status = False
        status = self.arm_move_group.go(pose_goal, wait = True)
        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()
        return status
        
if __name__ == '__main__':
    right_robot = Right_arm()
