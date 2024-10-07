#!/usr/bin/env python3

import rospy
import tf
import numpy as np
import image_geometry
import math
import cv2
from cv_bridge import CvBridge
from obj_segmentation.msg import SegmentedClustersArray
from toy_assembly.msg import ObjectImage
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Point
import sensor_msgs.point_cloud2 as pc2
from multiprocessing import Lock
import sys

class ImageSegment:
    def __init__(self):
        rospy.init_node('image_overlay', anonymous=True)
        self.bridge = CvBridge()
        
        self.listener = tf.TransformListener()

        self.real = rospy.get_param("~real", default=False)
        self.arm = rospy.get_param("~arm", default="left")

        self.workspace_depth = 0.75
        self.workspace_width = 0.5
        self.workspace_height = 0.5

        rospy.loginfo(f"real world:{self.real}")
        rospy.loginfo(f"arm: {self.arm}")
        '''
        <link name="${prefix}camera_depth_frame" />
        <joint name="${prefix}depth_module" type="fixed">
            <origin xyz="0.0275 0.066 -0.00305" rpy="3.14159265358979 3.14159265358979 0" />
            <parent link="${prefix}end_effector_link" />
            <child  link="${prefix}camera_depth_frame" />
        </joint>
        <link name="${prefix}camera_color_frame" />
        <joint name="${prefix}color_module" type="fixed">
            <origin xyz="0 0.05639 -0.00305" rpy="3.14159265358979 3.14159265358979 0" />
            <parent link="${prefix}end_effector_link" />
            <child  link="${prefix}camera_color_frame" />
        </joint>
        '''
        if self.real:
            self.cam_link_name = "left_camera_color_frame"
            cam_info_topic = f"/{self.arm}_camera/color/camera_info_throttled"
            rgb_image_topic = f"/{self.arm}_camera/color/image_rect_color_throttled"
            obj_cluster_topic = f"/{self.arm}_camera/depth_registered/object_clusters"
            output_image_topic = f"/{self.arm}_camera/color/overlay_raw"
        else:
            self.cam_link_name = "left_camera_link"
            cam_info_topic = f"/unity/camera/{self.arm}/rgb/camera_info"
            rgb_image_topic = f"/unity/camera/{self.arm}/rgb/image_raw"
            obj_cluster_topic = f"/unity/camera/{self.arm}/depth/object_clusters"
            output_image_topic = f"/unity/camera/{self.arm}/rgb/overlay_raw"


        rospy.loginfo(cam_info_topic)
        rospy.loginfo(rgb_image_topic)
        rospy.loginfo(obj_cluster_topic)

        self.buffer = 15

        self.mutex = Lock()
        self.objects = []

        self.rgb_cam_info = rospy.wait_for_message(cam_info_topic, CameraInfo, timeout=None)
        self.cam_model = image_geometry.PinholeCameraModel()
        self.cam_model.fromCameraInfo(self.rgb_cam_info)

        print(self.rgb_cam_info)

        self.frame = self.rgb_cam_info.header.frame_id
        print(self.frame)

        self.overlayed_images_pub = rospy.Publisher(output_image_topic, Image, queue_size=10)
        self.object_images_pub = rospy.Publisher(f'/{self.arm}_object_images', ObjectImage, queue_size=10)

        self.rgb_image_sub = rospy.Subscriber(rgb_image_topic, Image, self.image_cb)
        self.obj_cluster_sub = rospy.Subscriber(obj_cluster_topic, SegmentedClustersArray, self.cluster_callback)

        rospy.spin()

    def transform_points(self, center_point, min_point, max_point):
        t = rospy.Time(0)
        self.listener.waitForTransform(self.cam_link_name, self.frame, t, rospy.Duration(4.0))
        center_point = self.listener.transformPoint(self.frame, center_point)
        min_point = self.listener.transformPoint(self.frame, min_point)
        max_point = self.listener.transformPoint(self.frame, max_point)

        return center_point, min_point, max_point

    def cluster_callback(self, obj_clusters):
        with self.mutex:
            self.objects = []
            i = 0
            for pc in obj_clusters.clusters:
                #print("obj %d" % i)
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

                center_point = PointStamped()
                center_point.header = obj_clusters.header
                center_point.point.x = center[0]
                center_point.point.y = center[1]
                center_point.point.z = center[2]
                if self.check_in_workspace(center_point, i):
                    min_point = PointStamped()
                    min_point.header = obj_clusters.header
                    min_point.point.x = min_x
                    min_point.point.y = min_y
                    min_point.point.z = min_z

                    max_point = PointStamped()
                    max_point.header = obj_clusters.header
                    max_point.point.x = max_x
                    max_point.point.y = max_y
                    max_point.point.z = max_z

                    center_point, min_point, max_point = self.transform_points(center_point, min_point, max_point)

                    center = [center_point.point.x, center_point.point.y, center_point.point.z]
                    min_point = [min_point.point.x, min_point.point.y, min_point.point.z]
                    max_point = [max_point.point.x, max_point.point.y, max_point.point.z]

                    min_pix = self.cam_model.project3dToPixel( min_point )
                    #min_pix = (int(min_pix[0]), int(min_pix[0]))
                    max_pix = self.cam_model.project3dToPixel (max_point )
                    #max_pix = (int(max_pix[0]), int(max_pix[0]))
                    center_pix = self.cam_model.project3dToPixel( center )
                    #center_pix = (int(center_pix[0]), int(center_pix[0]))

                    obj = dict()
                    obj["i"] = i
                    obj["min_pix"] = min_pix
                    obj["max_pix"] = max_pix
                    obj["center_pix"] = center_pix
                    obj["center"] = center
                    self.objects.append(obj)
                    '''
                    rospy.loginfo(f"obj_{i}, {min_pix}, {center_pix}, {max_pix}")
                    print(min_x, max_x)
                    print(min_y, max_y)
                    print(min_z, max_z)
                    '''
                i += 1
            
            self.have_objects = True



    def image_cb(self, rgb_ros_image):
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_ros_image, desired_encoding="bgr8")
        rgb_img = rgb_img.copy()

        with self.mutex:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            line_type = cv2.LINE_AA

            for obj in self.objects:
                i = obj["i"]
                min_pix = obj["min_pix"]
                max_pix = obj["max_pix"]
                center_pix = obj["center_pix"]

                
                u_min = max(int(math.floor(min_pix[0]))-self.buffer, 0)
                v_min = max(int(math.floor(min_pix[1]))-self.buffer, 0)
                    
                u_max = min(int(math.ceil(max_pix[0]))+self.buffer, rgb_img.shape[1])
                v_max = min(int(math.ceil(max_pix[1]))+self.buffer, rgb_img.shape[1])

                cv2.rectangle(rgb_img, (u_min, v_min), (u_max, v_max), color=(255,255,255), thickness=2)

            object_positions = []
            for obj in self.objects:
                i = obj["i"]
                min_pix = obj["min_pix"]
                max_pix = obj["max_pix"]
                center_pix = obj["center_pix"]
                p = Point()
                p.x = obj["center"][0]
                p.y = obj["center"][1]
                p.z = obj["center"][2]
                object_positions.append(p)

                u_min = max(int(math.floor(min_pix[0]))-self.buffer, 0)
                v_min = max(int(math.floor(min_pix[1]))-self.buffer, 0)
                u_max = min(int(math.ceil(max_pix[0]))+self.buffer, rgb_img.shape[1])
                v_max = min(int(math.ceil(max_pix[1]))+self.buffer, rgb_img.shape[1])
                
                #rospy.loginfo(f"{i}, {center_pix}")

                #text_location = (int(center_pix[0]),int(center_pix[1]))  #center of object
                text_location = (u_min,v_min)   #top right rocer of object's bounding box
                text = "obj_"+str(i)
                
                cv2.putText(rgb_img, text, text_location, font, font_scale, (0,0,0), thickness+1, line_type)
                cv2.putText(rgb_img, text, text_location, font, font_scale, (255,255,255), thickness, line_type)

                #rospy.loginfo(text)
        
        '''
        ee = PointStamped()
        ee.header.frame_id = "right_tool_frame"

        ee_forward = PointStamped()
        ee_forward.header.frame_id = "right_tool_frame"
        ee_forward.point.x = 0.25

        ee_right = PointStamped()
        ee_right.header.frame_id = "right_tool_frame"
        ee_right.point.y = 0.25
        left = ()

        ee_up = PointStamped()
        ee_up.header.frame_id = "right_tool_frame"
        ee_up.point.z = 0.25
        

        t = rospy.Time(0)
        self.listener.waitForTransform(self.frame, "right_tool_frame",  t, rospy.Duration(4.0))

        ee = self.listener.transformPoint(self.frame, ee)
        ee_forward = self.listener.transformPoint(self.frame, ee_forward)
        ee_right = self.listener.transformPoint(self.frame, ee_right)
        ee_up = self.listener.transformPoint(self.frame, ee_up)

        ee = tuple(np.asarray(self.cam_model.project3dToPixel( [ee.point.x, ee.point.y, ee.point.z] ), int))
        ee_forward = tuple(np.asarray(self.cam_model.project3dToPixel( [ee_forward.point.x+self.offset_x, ee_forward.point.y+self.offset_y, ee_forward.point.z] ), int))
        ee_right = tuple(np.asarray(self.cam_model.project3dToPixel( [ee_right.point.x+self.offset_x, ee_right.point.y+self.offset_y, ee_right.point.z] ), int))
        ee_up = tuple(np.asarray(self.cam_model.project3dToPixel( [ee_up.point.x+self.offset_x, ee_up.point.y+self.offset_y, ee_up.point.z] ), int))
        
        cv2.line(rgb_img, ee, ee_forward, (255,0,0), 1) 
        cv2.line(rgb_img, ee, ee_right, (0,255,0), 1) 
        cv2.line(rgb_img, ee, ee_up, (0,0,255), 1)
        '''

        #rospy.loginfo(f"-----------------------")    
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_img, "bgr8")
        rgb_msg.header = rgb_ros_image.header
        rgb_msg.header.stamp = rospy.Time.now()
        self.overlayed_images_pub.publish(rgb_msg)

        object_image = ObjectImage()
        object_image.header = rgb_ros_image.header
        object_image.image = rgb_msg
        object_image.object_positions = object_positions
        self.object_images_pub.publish(object_image)

    def check_in_workspace(self, p, i):
        t = rospy.Time(0)
        self.listener.waitForTransform(p.header.frame_id, "right_base_link", t, rospy.Duration(4.0))
        p.header.stamp = t
        p = self.listener.transformPoint("right_base_link", p)
        check = (0.1 < p.point.x < self.workspace_depth) and (-self.workspace_width < p.point.y < self.workspace_width) and (p.point.z < self.workspace_height)

        if check:
            rospy.loginfo(f"cluster {i}, point [{p.point.x:.2f},{p.point.y:.2f},{p.point.z:.2f}] check: {check}")
            return True
        else:
            rospy.loginfo(f"cluster {i}, point [{p.point.x:.2f},{p.point.y:.2f},{p.point.z:.2f}] check: {check}")
            return False

     
        
if __name__ == '__main__':
    segmenter = ImageSegment()

