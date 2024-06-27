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

        self.real = rospy.get_param("real", default=False)
        self.arm = rospy.get_param("arm", default="left")


        rospy.loginfo(self.real)
        rospy.loginfo(self.arm)

        if self.real:
            cam_info_topic = f"/{self.arm}_camera/color/camera_info"
            rgb_image_topic = f"/{self.arm}_camera/color/image_raw"
            output_image_topic = f"/{self.arm}_camera/color/overlay_raw"
            #obj_cluster_topic = f"/{self.arm}_camera/depth_registered/object_clusters"
        else:
            cam_info_topic = f"/unity/camera/{self.arm}/rgb/camera_info"
            rgb_image_topic = f"/unity/camera/{self.arm}/rgb/image_raw"
            output_image_topic = f"/unity/camera/{self.arm}/rgb/overlay_raw"
            #obj_cluster_topic = f"/unity/camera/{self.arm}/depth/object_clusters"

        obj_cluster_topic = "output_cloud"

        rospy.loginfo(cam_info_topic)
        rospy.loginfo(rgb_image_topic)
        rospy.loginfo(obj_cluster_topic)

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
        t = rospy.Time.now()
        self.listener.waitForTransform('left_camera_link', self.frame, t, rospy.Duration(4.0))
        center_point = self.listener.transformPoint(self.frame, center_point)
        min_point = self.listener.transformPoint(self.frame, min_point)
        max_point = self.listener.transformPoint(self.frame, max_point)

        return center_point, min_point, max_point

    def cluster_callback(self, obj_clusters):
        with self.mutex:
            self.objects = []
            for i, pc in enumerate(obj_clusters.clusters):
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
                max_pix = self.cam_model.project3dToPixel (max_point )
                center_pix = self.cam_model.project3dToPixel( center )

                obj = dict()
                obj["i"] = i
                obj["min_pix"] = min_pix
                obj["max_pix"] = max_pix
                obj["center_pix"] = center_pix
                obj["center"] = center
                self.objects.append(obj)
                #rospy.loginfo(f"{i}, {center_pix}")
                #print(min_x, max_x)
                #print(min_y, max_y)
                #print(min_z, max_z)
            
            self.have_objects = True



    def image_cb(self, rgb_ros_image):
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_ros_image, desired_encoding="passthrough")
        rgb_img = rgb_img.copy()

        with self.mutex:
            if len(self.objects) < 1:
                return

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            line_type = cv2.LINE_AA

            for obj in self.objects:
                i = obj["i"]
                min_pix = obj["min_pix"]
                max_pix = obj["max_pix"]
                center_pix = obj["center_pix"]

                buffer = 5
                u_min = max(int(math.floor(min_pix[0]))-buffer, 0)
                v_min = max(int(math.floor(min_pix[1]))-buffer, 0)
                    
                u_max = min(int(math.ceil(max_pix[0]))+buffer, rgb_img.shape[1])
                v_max = min(int(math.ceil(max_pix[1]))+buffer, rgb_img.shape[1])

                cv2.rectangle(rgb_img, (u_min, v_min), (u_max, v_max), color=(255,255,255), thickness=1)

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

                u_min = max(int(math.floor(min_pix[0])), 0)
                v_min = max(int(math.floor(min_pix[1])), 0)
                    
                u_max = min(int(math.ceil(max_pix[0])), rgb_img.shape[1])
                v_max = min(int(math.ceil(max_pix[1])), rgb_img.shape[1])
                
                #rospy.loginfo(f"{i}, {center_pix}")

                #text_location = (int(center_pix[0]),int(center_pix[1]))  #center of object
                text_location = (u_min,v_min)   #top right rocer of object's bounding box
                text = "obj_"+str(i)
                
                cv2.putText(rgb_img, text, text_location, font, font_scale, (0,0,0), thickness+1, line_type)
                cv2.putText(rgb_img, text, text_location, font, font_scale, (255,255,255), thickness, line_type)

                #rospy.loginfo(text)
        


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


        
if __name__ == '__main__':
    segmenter = ImageSegment()

