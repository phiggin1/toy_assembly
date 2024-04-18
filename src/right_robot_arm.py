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

        self.start_pose = [0.00062, -0.12636, -1.07251, -0.00121, -2.11940, 1.57205]
        self.init_position()	

        #self.cam_info = rospy.wait_for_message('/unity/camera/right/rgb/camera_info', CameraInfo)
        #self.cam_model = PinholeCameraModel()
        #self.cam_model.fromCameraInfo(self.cam_info)
        
        self.horse_topic = '/object_positions'
        self.horse_pose = None  
        #self.horses = rospy.Subscriber(self.horse_topic, PoseStamped, self.get_horse_pose)
        #self.horse_pose = []

        self.listener = tf.TransformListener()
        self.grab = rospy.Publisher('/buttons', String, queue_size=10)

        self.move_pose = rospy.Service('move_pose', MoveITPose, self.move_pose)

        self.grab_object = rospy.Service('grab_object', MoveITGrabPose, self.get_object)
        rospy.spin()
	
    def init_position(self):
        moveit_commander.roscpp_initialize(sys.argv)
        #rospy.init_node("move_to_start", anonymous=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group_name = "arm"
        self.arm_move_group = moveit_commander.MoveGroupCommander(self.arm_group_name)
        self.planning_frame = "right_base_link"

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

    def move_pose(self, request):
        object_pose = self.transform_obj_pos(request.pose)        
        #print(object_pose)
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
        a = dict()
        a["robot"] = "right"
        a["action"] = "grab"
        s = json.dumps(a)
        self.grab.publish(s)


        self.arm_move_group.set_max_velocity_scaling_factor(0.750)
        self.arm_move_group.go(self.start_pose, wait=True) 
        self.arm_move_group.stop()

        return status
        
        
    def transform_obj_pos(self, obj_pos):
        t = rospy.Time.now()
        obj_pos.header.stamp = t
        self.listener.waitForTransform(obj_pos.header.frame_id, self.planning_frame, t, rospy.Duration(4.0))
        obj_pos = self.listener.transformPose(self.planning_frame, obj_pos)
        return obj_pos
	
        
if __name__ == '__main__':
    right_robot = Right_arm()
