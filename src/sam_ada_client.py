#!/usr/bin/env python3

import zmq
import tf
import numpy as np
import rospy
import message_filters
from cv_bridge import CvBridge
from toy_assembly.srv import SAM, SAMResponse
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Point
from image_geometry import PinholeCameraModel
from multiprocessing import Lock
from toy_assembly.msg import Detection
import pycocotools.mask as mask_util
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct
import json

class SamAdaClient:
    def __init__(self):
        rospy.init_node('SamAdaClient')

        self.mutex = Lock()

        self.listener = tf.TransformListener()
        self.cvbridge = CvBridge()
        
        self.debug = rospy.get_param("~debug", True)
        server_port = rospy.get_param("~port", "8899")

        self.real = rospy.get_param("~real", default=False)
        self.arm = rospy.get_param("~arm", default="left")

        rospy.loginfo(f"debug:{self.debug}")
        rospy.loginfo(f"server_port: {server_port}")
        rospy.loginfo(f"real world:{self.real}")
        rospy.loginfo(f"arm: {self.arm}")

        if self.real:
            self.cam_link_name = f"{self.arm}_camera_color_frame"
            cam_info_topic = f"/{self.arm}_camera/color/camera_info_throttled"
            rgb_image_topic = f"/{self.arm}_camera/color/image_rect_color_throttled"              
            depth_image_topic = f"/{self.arm}_camera/depth_registered/sw_registered/image_rect_throttled"
            output_image_topic = f"/{self.arm}_camera/color/sam_overlay_raw"
        else:
            self.cam_link_name = f"{self.arm}_camera_link"
            cam_info_topic = f"/unity/camera/{self.arm}/rgb/camera_info"
            rgb_image_topic = f"/unity/camera/{self.arm}/rgb/image_raw"
            depth_image_topic = f"/unity/camera/{self.arm}/depth/image_raw"
            output_image_topic = f"/unity/camera/{self.arm}/rgb/sam_overlay_raw"

        rospy.loginfo(cam_info_topic)
        rospy.loginfo(rgb_image_topic)
        rospy.loginfo(depth_image_topic)

        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % server_port)
        rospy.loginfo(f"Server listening on port:{server_port}")

        self.cam_info = rospy.wait_for_message(cam_info_topic, CameraInfo)
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(self.cam_info)

        self.rgb_sub = message_filters.Subscriber(rgb_image_topic, Image)
        self.depth_sub = message_filters.Subscriber(depth_image_topic, Image)
        ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], slop=0.2, queue_size=10)
        ts.registerCallback(self.image_cb)
        self.rgb_image = None
        self.depth_image = None
        self.have_images = False

        rate = rospy.Rate(5)
        while not self.have_images:
            rate.sleep()
        
        self.sam_serv = rospy.Service('/get_sam_segmentation', SAM, self.SAM)
        self.annote_pub = rospy.Publisher(output_image_topic, Image, queue_size=10)

        rospy.spin()

    def image_cb(self, rgb_ros_image, depth_ros_image):
        success = self.mutex.acquire(block=False, timeout=0.02)
        if success:
            self.rgb_encoding = rgb_ros_image.encoding
            self.depth_encoding = depth_ros_image.encoding
            self.rgb_image = self.cvbridge.imgmsg_to_cv2(rgb_ros_image, desired_encoding="bgr8") 
            self.depth_image = self.cvbridge.imgmsg_to_cv2(depth_ros_image)#, desired_encoding="passthrough") 
            if not self.have_images:
                rospy.loginfo("HAVE IMAGES")
                print(f"rgb eoncoding: {self.rgb_encoding}")
                print(f"depth eoncoding: {self.depth_encoding }")
                self.have_images = True
            self.mutex.release()

    def SAM(self, request):
        if self.debug: rospy.loginfo('SAM req recv')
        with self.mutex:
            if len(request.image.data) < 10:
                image = self.rgb_image
            else:
                image = self.cvbridge.imgmsg_to_cv2(request.image, desired_encoding="bgr8") 
            text = request.text_prompt

            print(text)
            self.colors = {
                "tan tray":(255,0,0),
                "orange tray":(0,255,0),
                "tan horse body":(0,0,255),
                "blue horse legs":(255,255,0),
                "orange horse legs":(255,0,255),
                "table":(255,255,255)
            }

            print(image.shape)

            msg = {"type":"sam",
                "image":image.tolist(),
                "text":text,
            }

            print(msg["text"])

            if self.debug: rospy.loginfo("SAM sending to ada")
            
            self.socket.send_json(msg)
            resp = self.socket.recv_json()
            if self.debug: rospy.loginfo('SAM recv from ada') 

            response = SAMResponse()
            response.annotated_image = self.cvbridge.cv2_to_imgmsg(np.asarray(resp['annotated_image'], dtype=np.uint8), encoding="bgr8")
            response.annotated_image.header.frame_id = self.cam_info.header.frame_id
            response.annotated_image.header.stamp = rospy.Time.now()

            self.annote_pub.publish(response.annotated_image)

            self.points = []
            for detection in resp["annotations"]:
                rospy.loginfo(f"{detection['class_name']}, {detection['score']}, {detection['bbox']}")
                obj = Detection()

                obj.class_name = detection['class_name']

                obj.u_min = int(detection['bbox'][0])
                obj.v_min = int(detection['bbox'][1])

                obj.u_max = int(detection['bbox'][2])
                obj.v_max = int(detection['bbox'][3])

                if isinstance(detection['score'], float):
                    obj.score = detection['score']
                else:
                    obj.score = detection['score'][0]

                obj.rle_encoded_mask = json.dumps(detection['segmentation'])
                
                response.object.append(obj)

            rospy.loginfo('reply')
            return response



if __name__ == '__main__':
    sam = SamAdaClient()

