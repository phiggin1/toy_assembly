#!/usr/bin/env python3

import sys
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from image_geometry import PinholeCameraModel
import sensor_msgs.point_cloud2 as pc2
from toy_assembly.srv import SAM
from toy_assembly.srv import CLIP
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from obj_segmentation.msg import SegmentedClustersArray


class Left_arm:
    def __init__(self):
        rospy.init_node("leftArm")

        self.start_pose = [-0.88708, -0.01635, -2.04250, -2.17897, 1.67276, 2.33558]
        self.goal_tolerance = 0.01
        self.init_position()

        self.cam_info = rospy.wait_for_message('/unity/camera/left/depth/camera_info', CameraInfo)
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(self.cam_info)

        self.cluster_topic = "/unity/camera/left/depth/filtered_object_clusters"
        self.clusters = rospy.wait_for_message(self.cluster_topic, SegmentedClustersArray)

        self.horses_pose = []

        self.horse_pose_pub = rospy.Publisher('/object_positions', PoseStamped, queue_size=10)  # may need to edit topics

        # pose array has a specific format but i can make those zeros so dont have to worry about setting up the message, hence just sending the position

        while not rospy.is_shutdown():
            self.get_horses_pose(self.clusters);
        
    def init_position(self):
        moveit_commander.roscpp_initialize(sys.argv)
        #rospy.init_node("move_to_start", anonymous=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group_name = "arm"
        self.arm_move_group = moveit_commander.MoveGroupCommander(self.arm_group_name)
        self.planning_frame = self.arm_move_group.get_planning_frame()

        self.arm_move_group.set_max_velocity_scaling_factor(0.750)
        self.arm_move_group.set_goal_joint_tolerance(self.goal_tolerance)
        self.arm_move_group.go(self.start_pose, wait=True)       
        self.arm_move_group.stop()

    def get_horses_pose(self, clusters):
        for i, pc in enumerate(clusters.clusters):
            #print(f"obj {i}")
            p = self.get_centroid(pc)
            self.horses_pose.append(p)

        """
        dont need to convert to pixel here. send locations as is under the 3d frame of reference of the left arm to the right arm, 
        where you can use the header message to then change the frame id to that of the right arm and move from there.
        """
        first_obj_pos = PoseStamped()
        first_obj_pos.header.frame_id = clusters.header.frame_id
        first_obj_pos.pose.position.x = self.horses_pose[0][0]
        first_obj_pos.pose.position.y = self.horses_pose[0][1]
        first_obj_pos.pose.position.z = self.horses_pose[0][2]
        first_obj_pos.header.stamp = rospy.Time.now()
        self.horse_pose_pub.publish(first_obj_pos)

    def get_centroid(self, pc):
        min_x = 1000.0        
        min_y = 1000.0
        min_z = 1000.0
        max_x = -1000.0
        max_y = -1000.0
        max_z = -1000.0

        #for each object get a bounding box
        for p in pc2.read_points(pc):
            if p[0] > max_x:
                max_x = p[0]
            if p[0] < min_x:
                min_x = p[0]

            if p[1] > max_y:
                max_y = p[1]
            if p[1] < min_y:
                min_y = p[1]

            if p[2] > max_z:
                max_z = p[2]
            if p[2] < min_z:
                min_z = p[2]

        center = [(min_x + max_x)/2, (min_y + max_y)/2, (min_z + max_z)/2]
        w = max_x-min_x
        h = max_y-min_y
        d = max_z-min_z

        return center


if __name__ == '__main__':
    left_robot = Left_arm()
