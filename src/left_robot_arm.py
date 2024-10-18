#!/usr/bin/env python3

import sys
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


class Left_arm:
    def __init__(self):
        rospy.init_node("leftArm")
        self.listener = tf.TransformListener()

        self.base_frame = "world"
        self.arm = "left"
        self.padding = 0.075
        self.frames = [
            "left_base_link",
            "left_shoulder_link",
            "left_bicep_link",
            "left_forearm_link",
            "left_spherical_wrist_1_link",
            "left_spherical_wrist_2_link",
            "left_bracelet_link",
            "left_end_effector_link",
            "left_tool_frame"
        ]

        self.start_pose = [-0.71, -0.65, 2.22, -1.37, 1.30, 1.40]

        self.goal_tolerance = 0.01
        self.init_position()

        #self.cam_info = rospy.wait_for_message('/unity/camera/left/depth/camera_info', CameraInfo)
        #self.cam_model = PinholeCameraModel()
        #self.cam_model.fromCameraInfo(self.cam_info)

        #self.cluster_topic = "/unity/camera/left/depth/filtered_object_clusters"
        #self.clusters = rospy.wait_for_message(self.cluster_topic, SegmentedClustersArray)

        self.horses_pose = []

        self.horse_pose_pub = rospy.Publisher('/object_positions', PoseStamped, queue_size=10)  # may need to edit topics

        # pose array has a specific format but i can make those zeros so dont have to worry about setting up the message, hence just sending the position
        #rospy.spin()

        #while not rospy.is_shutdown():
        #    self.get_horses_pose(self.clusters);

        rate = rospy.Rate(5)
        self.marker_pub = rospy.Publisher(f"{self.arm}_arm_bbox", Marker, queue_size=10)
        while not rospy.is_shutdown():
            marker = self.get_marker(self.arm)
            self.marker_pub.publish(marker)
            rate.sleep()
        
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


    def get_marker(self, arm):
        points = []
        target = PointStamped()
        for frame in self.frames:
            self.listener.waitForTransform(frame, self.base_frame, rospy.Time(), rospy.Duration(4.0) )
            target.header.frame_id = frame
            target.header.stamp = rospy.Time()
            transformned_target = self.listener.transformPoint(self.base_frame, target)
            points.append((transformned_target.point,frame))

        marker = self.get_bbox(points)
        
        return marker
        
    def get_bbox(self, points):
        min_x =  99.9
        max_x = -99.9
        min_y =  99.9
        max_y = -99.9
        min_z =  99.9
        max_z = -99.9
        for p, frame in points:
            #print(f"{frame}: {p.x}, {p.y}, {p.z}")
            if p.x > max_x:
                max_x = p.x
            if p.x < min_x:
                min_x = p.x

            if p.y > max_y:
                max_y = p.y
            if p.y < min_y:
                min_y = p.y

            if p.z > max_z:
                max_z = p.z
            if p.z < min_z:
                min_z = p.z

        min_x -= self.padding
        max_x += self.padding
        min_y -= self.padding
        max_y += self.padding
        min_z -= self.padding
        max_z += self.padding

        #print(f"{self.arm}: {min_x:,.4f}, {max_x:,.4f}, {min_y:,.4f}, {max_y:,.4f}, {min_z:,.4f}, {max_z:,.4f}")

        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.type = Marker.CUBE
        marker.pose.position.x = (max_x + min_x)/2.0
        marker.pose.position.y = (max_y + min_y)/2.0
        marker.pose.position.z = (max_z + min_z)/2.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = (max_x - min_x)
        marker.scale.y = (max_y - min_y)
        marker.scale.z = (max_z - min_z)

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.25

        return marker

if __name__ == '__main__':
    left_robot = Left_arm()
