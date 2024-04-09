#!/usr/bin/env python3

import rospy
import message_filters
import numpy as np
import image_geometry
import math
import cv2
from cv_bridge import CvBridge
from obj_segmentation.msg import SegmentedClustersArray

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from multiprocessing import Lock

import sys


class ImageSegment:
    def __init__(self):
        rospy.init_node('image_segment', anonymous=True)
        self.bridge = CvBridge()

        self.mutex = Lock()
        self.objects = []

        self.rgb_cam_info = rospy.wait_for_message("/unity/camera/left/rgb/camera_info", CameraInfo, timeout=None)
        self.cam_model = image_geometry.PinholeCameraModel()
        self.cam_model.fromCameraInfo(self.rgb_cam_info)

        print(self.rgb_cam_info)

        self.overlayed_images_pub = rospy.Publisher('/overlayed_images', Image, queue_size=10)

    
        self.rgb_image_sub = rospy.Subscriber('/unity/camera/left/rgb/image_raw', Image, self.image_cb)
        self.obj_cluster_sub = rospy.Subscriber("/filtered_object_clusters", SegmentedClustersArray, self.cluster_callback)
        

        '''
        self.rgb_image_sub = message_filters.Subscriber("/unity/camera/left/rgb/image_raw", Image)
        self.obj_cluster_sub = message_filters.Subscriber("/filtered_object_clusters", SegmentedClustersArray)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_image_sub, self.obj_cluster_sub], queue_size=10, slop=12.0, allow_headerless=True)
        self.ts.registerCallback(self.callback)
        '''

        rospy.spin()

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

                min_pix = self.cam_model.project3dToPixel( [ min_x, min_y, min_z ] )
                max_pix = self.cam_model.project3dToPixel( [ max_x, max_y, max_z ] )
                center_pix = self.cam_model.project3dToPixel( center )

                obj = dict()
                obj["i"] = i
                obj["min_pix"] = min_pix
                obj["max_pix"] = max_pix
                obj["center_pix"] = center_pix
                self.objects.append(obj)

                rospy.loginfo(f"{i}, {center_pix}")
            
            self.have_objects = True



    def image_cb(self, rgb_ros_image):
        rgb_img = np.asarray(self.bridge.imgmsg_to_cv2(rgb_ros_image, desired_encoding="passthrough"))

        with self.mutex:
            if len(self.objects) < 1:
                return

            for obj in self.objects:
                i = obj["i"]
                min_pix = obj["min_pix"]
                max_pix = obj["max_pix"]
                center_pix = obj["center_pix"]

                #25 is to try and avoid clipping the object off
                #this might not be needed/a bad idea
                u_min = max(int(math.floor(min_pix[0]))-25, 0)
                v_min = max(int(math.floor(min_pix[1]))-25, 0)
                    
                u_max = min(int(math.ceil(max_pix[0]))+25, rgb_img.shape[1])
                v_max = min(int(math.ceil(max_pix[1]))+25, rgb_img.shape[1])
                
                rospy.loginfo(f"{i}, {center_pix}")
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (int(center_pix[0]),int(center_pix[1]))#(u_min,v_min)
                fontScale = 0.75
                fontColor = (255,255,255)
                thickness = 2
                lineType = cv2.LINE_AA
                text = "obj_"+str(i)

                #mask = cv2.putText(mask, "PLOTCH", (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 5)
                #mask = cv2.putText(mask, "PLOTCH", (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2)

                cv2.putText(rgb_img, text, bottomLeftCornerOfText, font, fontScale, (0,0,0), thickness+3, lineType)
                cv2.putText(rgb_img, text, bottomLeftCornerOfText, font, fontScale, (255,255,255), thickness, lineType)
                rospy.loginfo(text)
            
        rospy.loginfo(f"-----------------------")    
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_img, "bgr8")
        self.overlayed_images_pub.publish(rgb_msg)

if __name__ == '__main__':
    segmenter = ImageSegment()

