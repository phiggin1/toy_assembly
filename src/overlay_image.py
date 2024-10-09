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
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from visualization_msgs.msg import Marker

def get_marker(frame_id, stamp, id, x, y ,z, w ,d ,h):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.type = Marker.CUBE
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = z
    m.pose.orientation.w = 1.0
    m.scale.x = d
    m.scale.y = w
    m.scale.z = h
    m.color.r = 1.0
    m.color.g = 1.0
    m.color.b = 1.0
    m.color.a = 0.25

    return m
    
class ImageSegment:
    def __init__(self):
        rospy.init_node('image_overlay', anonymous=True)
        self.bridge = CvBridge()
        
        self.obj_marker_pub = rospy.Publisher("/obj_marker_text", Marker, queue_size=10)
        self.min_pt_pub = rospy.Publisher("/zz/min_pt", PointStamped, queue_size=10)
        self.max_pt_pub = rospy.Publisher("/zz/max_pt", PointStamped, queue_size=10)

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
            self.cam_link_name = f"{self.arm}_camera_color_frame"
            cam_info_topic = f"/{self.arm}_camera/color/camera_info_throttled"
            rgb_image_topic = f"/{self.arm}_camera/color/image_rect_color_throttled"
            obj_cluster_topic = f"/{self.arm}_camera/depth_registered/object_clusters"
            output_image_topic = f"/{self.arm}_camera/color/overlay_raw"
        else:
            self.cam_link_name = f"{self.arm}_camera_link"
            cam_info_topic = f"/unity/camera/{self.arm}/rgb/camera_info"
            rgb_image_topic = f"/unity/camera/{self.arm}/rgb/image_raw"
            obj_cluster_topic = f"/unity/camera/left/depth/object_clusters"
            output_image_topic = f"/unity/camera/{self.arm}/rgb/overlay_raw"


        rospy.loginfo(cam_info_topic)
        rospy.loginfo(rgb_image_topic)
        rospy.loginfo(obj_cluster_topic)

        self.buffer = 0#15

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

    def get_bounding_points(self, min_point, max_point, frame):
        min_x = 1000.0
        min_y = 1000.0
        min_z = 1000.0
        max_x = -1000.0
        max_y = -1000.0
        max_z = -1000.0
        pts = [
            [min_point.point.x,min_point.point.y,min_point.point.z],
            [max_point.point.x,min_point.point.y,min_point.point.z],
            [min_point.point.x,min_point.point.y,max_point.point.z],
            [max_point.point.x,min_point.point.y,max_point.point.z],
            [min_point.point.x,max_point.point.y,min_point.point.z],
            [max_point.point.x,max_point.point.y,min_point.point.z],
            [min_point.point.x,max_point.point.y,max_point.point.z],
            [max_point.point.x,max_point.point.y,max_point.point.z]
        ]

        self.listener.waitForTransform(source_frame=frame, target_frame=self.frame, time=rospy.Time(0), timeout=rospy.Duration(4.0))
        for p in pts:
            point = PointStamped()
            point.header = min_point.header
            point.point.x = p[0]
            point.point.y = p[1]
            point.point.z = p[2]

            p_new = self.listener.transformPoint(self.frame, point)
            if p_new.point.x > max_x:
                max_x = p_new.point.x
            if p_new.point.x < min_x:
                min_x = p_new.point.x

            if p_new.point.y > max_y:
                max_y = p_new.point.y
            if p_new.point.y < min_y:
                min_y = p_new.point.y

            if p_new.point.z > max_z:
                max_z = p_new.point.z
            if p_new.point.z < min_z:
                min_z = p_new.point.z

        new_min = PointStamped()
        new_min.header.frame_id = self.frame
        new_min.point.x = min_x
        new_min.point.y = max_y
        new_min.point.z = min_z

        new_max = PointStamped()
        new_max.header.frame_id = self.frame
        new_max.point.x = max_x
        new_max.point.y = min_y
        new_max.point.z = max_z


        return new_min, new_max

    def cluster_callback(self, obj_clusters):
        with self.mutex:
            self.objects = []
            i = 0
            for pc in obj_clusters.clusters:
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
                center_point.header.frame_id = obj_clusters.header.frame_id
                center_point.point.x = center[0]
                center_point.point.y = center[1]
                center_point.point.z = center[2]

                if self.check_in_workspace(center_point, i):
                    #print('=========')
                    #rospy.loginfo(f"obj_{i}, {min_x}, {min_y}, {min_z}")
                    #rospy.loginfo(f"obj_{i}, {max_x}, {max_y}, {max_z}")
                    min_point = PointStamped()
                    min_point.header.frame_id = obj_clusters.header.frame_id
                    min_point.point.x = min_x
                    min_point.point.y = min_y
                    min_point.point.z = min_z

                    max_point = PointStamped()
                    max_point.header.frame_id = obj_clusters.header.frame_id
                    max_point.point.x = max_x
                    max_point.point.y = max_y
                    max_point.point.z = max_z

                    min_point, max_point = self.get_bounding_points(min_point, max_point, obj_clusters.header.frame_id)

                    center = [min_point.point.x, (min_point.point.y+max_point.point.y)/2, (min_point.point.z+max_point.point.z)/2]
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
                    obj["center"] = center_point
                    self.objects.append(obj)
                    
                    #rospy.loginfo(f"{self.arm}, obj_{i}, {min_pix}, {center_pix}, {max_pix}")
                    #self.obj_marker_pub.publish( get_marker('world', rospy.Time.now(), 1, (max_x + min_x)/2, (max_y + min_y)/2, (max_z + min_z)/2, max_y - min_y ,max_x - min_x, max_z - min_z) )
                    
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
                object_positions.append(obj["center"])

                u_min = max(int(math.floor(min_pix[0]))-self.buffer, 0)
                v_min = max(int(math.floor(min_pix[1]))-self.buffer, 0)
                u_max = min(int(math.ceil(max_pix[0]))+self.buffer, rgb_img.shape[1])
                v_max = min(int(math.ceil(max_pix[1]))+self.buffer, rgb_img.shape[1])
                
                #rospy.loginfo(f"{i}, {center_pix}")

                #text_location = (int(center_pix[0]),int(center_pix[1]))  #center of object
                text_location = (u_min,v_max)   #top right corner of object's bounding box
                text = "obj_"+str(i)
                
                cv2.putText(rgb_img, text, text_location, font, font_scale, (0,0,0), thickness+1, line_type)
                cv2.putText(rgb_img, text, text_location, font, font_scale, (255,255,255), thickness, line_type)

                #rospy.loginfo(text)
        
        
        length = 0.05
        origin = -0.0
        ee = PointStamped()
        ee.header.frame_id = "right_tool_frame"
        ee.point.x = origin

        ee_forward = PointStamped()
        ee_forward.header.frame_id = "right_tool_frame"
        ee_forward.point.z = length

        ee_right = PointStamped()
        ee_right.header.frame_id = "right_tool_frame"
        ee_right.point.x = length

        ee_up = PointStamped()
        ee_up.header.frame_id = "right_tool_frame"
        ee_up.point.y = length
        

        t = rospy.Time(0)
        self.listener.waitForTransform(self.frame, "right_tool_frame",  t, rospy.Duration(4.0))

        ee = self.listener.transformPoint(self.frame, ee)
        ee_forward = self.listener.transformPoint(self.frame, ee_forward)
        ee_right = self.listener.transformPoint(self.frame, ee_right)
        ee_up = self.listener.transformPoint(self.frame, ee_up)

        ee = tuple(np.asarray(self.cam_model.project3dToPixel( [ee.point.x, ee.point.y, ee.point.z] ), int))
        ee_forward = tuple(np.asarray(self.cam_model.project3dToPixel( [ee_forward.point.x, ee_forward.point.y, ee_forward.point.z] ), int))
        ee_right = tuple(np.asarray(self.cam_model.project3dToPixel( [ee_right.point.x, ee_right.point.y, ee_right.point.z] ), int))
        ee_up = tuple(np.asarray(self.cam_model.project3dToPixel( [ee_up.point.x, ee_up.point.y, ee_up.point.z] ), int))
        blue = (255, 0 ,0)
        green = (0, 255, 0)
        red = (0, 0 ,255)
        #color is in BGR
        cv2.line(rgb_img, ee, ee_forward, blue, 2) 
        cv2.line(rgb_img, ee, ee_right, red, 2) 
        cv2.line(rgb_img, ee, ee_up, green, 2)
        
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
        self.listener.waitForTransform(source_frame=p.header.frame_id, target_frame="right_base_link", time=rospy.Time(0), timeout=rospy.Duration(4.0))
        p_right = self.listener.transformPoint("right_base_link", p)
        check = (0.1 < p_right.point.x < self.workspace_depth) and (-self.workspace_width < p_right.point.y < self.workspace_width) and (p_right.point.z < self.workspace_height)

        if check:
            #rospy.loginfo(f"cluster {i}, point [{p_right.point.x:.2f},{p_right.point.y:.2f},{p_right.point.z:.2f}] check: {check}")
            return True
        else:
            #rospy.loginfo(f"cluster {i}, point [{p.point.x:.2f},{p.point.y:.2f},{p.point.z:.2f}] check: {check}")
            return False

     
        
if __name__ == '__main__':
    segmenter = ImageSegment()

