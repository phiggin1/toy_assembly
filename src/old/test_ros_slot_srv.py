#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from toy_assembly.srv import DetectSlot, DetectSlotRequest

rospy.init_node('test_slot', anonymous=True)

cam_info_topic =    "/unity/camera/rgb/camera_info"
rgb_image_topic =   "/unity/camera/rgb/image_raw"

rospy.loginfo(f"cam_info_topic:{cam_info_topic}")
rospy.loginfo(f"rgb_image_topic:{rgb_image_topic}")
              
rgb_image = rospy.wait_for_message(rgb_image_topic, Image) 
rospy.loginfo("Got RGB image")

cam_info = rospy.wait_for_message(cam_info_topic, CameraInfo)
rospy.loginfo("Got cam_info")

location = PointStamped()
location.header = rgb_image.header
location.point.x = 0.0
location.point.y = 0.0
location.point.z = 1.0

detect_slot_serv =  rospy.ServiceProxy('get_slot_location', DetectSlot)

req = DetectSlotRequest()

req.rgb_image = rgb_image
req.depth_image = rgb_image
req.cam_info = cam_info
req.location = location

resp = detect_slot_serv(req)

rospy.loginfo(resp)