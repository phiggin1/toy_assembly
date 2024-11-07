#!/usr/bin/env python3

import sys
import math
import rospy
import numpy as np
import cv2
import tf
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
from geometry_msgs.msg import PoseStamped, PointStamped
from geometry_msgs.msg import Pose
from obj_segmentation.msg import SegmentedClustersArray
from visualization_msgs.msg import Marker, MarkerArray

# Publish .pcd data from command line

class Test_Find_Orientation:
    def __init__(self):
        rospy.init_node('test_orientation')
        self.obj_image = rospy.Subscriber("/cloud_pcd", PointCloud2, self.find_orientation)

        self.gripper_orientation = {'hand_pointing_right_cam_up': [math.sqrt(2)/2, 0, 0, math.sqrt(2)/2],
                    'hand_pointing_right_cam_front': [0.5, 0.5, -0.5, 0.5],
                    'hand_pointing_left_cam_up': [0, math.sqrt(2)/2, math.sqrt(2)/2, 0],
                    'hand_pointing_left_cam_front': [-0.5, -0.5, -0.5, -0.5],
                    'hand_pointing_down_cam_right': [-1, 0, 0, 0],
                    'hand_pointing_down_cam_front': [-math.sqrt(2)/2, -math.sqrt(2)/2, 0, 0],
                    'hand_pointing_forward_cam_up': [0.5, 0.5, 0.5, 0.5],
                    'hand_pointing_forward_cam_right' : [math.sqrt(2)/2, 0, math.sqrt(2)/2, 0]}

        self.marker_pub = rospy.Publisher("obj_marker", Marker, queue_size=10)
        self.obj_marker = Marker()
        self.obj_marker.header.frame_id = 'world'
        self.obj_marker.type = Marker.CUBE
        self.obj_marker.color.r = 0.0
        self.obj_marker.color.g = 1.0
        self.obj_marker.color.b = 1.0
        self.obj_marker.color.a = 1.0

        rospy.spin()        
    """
    Threshold approach
    """

    def find_orientation(self, msg):
        threshold = 0.6   #change val
        width, height, depth = self.get_width(msg)
        ratio = depth/width

        if ratio > threshold:
            object_orientation = 'orthogonal'
            #print('orthogonal')
            return 'hand_pointing_down_cam_right'
        else:
            object_orientation = 'parallel'
            #print('parallel')
            return 'hand_pointing_down_cam_front'
        

    def get_width(self, msg):    
        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        min_z = float('inf')
        max_z = float('-inf')
        
        for point in points:
            x = point[0]
            y = point[1]
            z = point[2] 
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            if z < min_z:
                min_z = z
            if z > max_z:
                max_z = z
        
        width = max_x - min_x
        height = max_y - min_y
        depth = max_z - min_z

        self.obj_marker.pose.position.x = (max_x + min_x) / 2
        self.obj_marker.pose.position.y = (max_y + min_y) / 2
        self.obj_marker.pose.position.z = (max_z + min_z) /2

        self.obj_marker.scale.x = max_x - min_x
        self.obj_marker.scale.y = max_y - min_y
        self.obj_marker.scale.z = max_z - min_z

        self.marker_pub.publish(self.obj_marker)

        return width, height, depth
    """        
    def get_height(self):
        points = pc2.read_points(self.obj_image, field_names=("x", "y", "z"), skip_nans=True)
        
        min_y = float('inf')
        max_y = float('-inf')
        
        for point in points:
            y = point[1] 
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
        
        height = max_y - min_y
        print(f'Height: {height}')
        return height

    def get_depth(self):
        points = pc2.read_points(self.obj_image, field_names=("x", "y", "z"), skip_nans=True)
        
        min_z = float('inf')
        max_z = float('-inf')
        
        for point in points:
            z = point[2] 
            if z < min_z:
                min_z = z
            if z > max_z:
                max_z = z
        
        depth = max_z - min_z
        print(f'Depth: {depth}')
        return depth

    def fit_to_data(self, training_data_p, training_data_o):
        self.parallel = [norm.fit(training_data_p)] # mu then sigma
        self.orthogonal = [norm.fit(training_data_o)]

    def classify_object_orientation_MLE(self):
        width = self.get_width()
        likelihood_parallel = norm.pdf(width, self.parallel[0], self.parallel[1])
        likelihood_orthogonal = norm.pdf(width, self.orthogonal[0], self.orthogonal[1])
        
        if likelihood_parallel > likelihood_orthogonal:
            object_orientation = 'parallel'
            return 'hand_pointing_down_cam_front'
        else:
            object_orientation = 'orthogonal'
            return 'hand_pointing_down_cam_right'
    """
if __name__ == '__main__':
    newTest = Test_Find_Orientation()
    #newTest.find_orientation()

