#!/usr/bin/env python3

import os
import rospy
import rosbag
import argparse
from sensor_msgs.msg import Image, CameraInfo, JointState 
from toy_assembly.msg import Transcription, ObjectImage
import cv2
from cv_bridge import CvBridge

class ExtractData:
    def __init__(self):
        rospy.init_node('extract_text', anonymous=True)
        self.bridge = CvBridge()
        path = "/mnt/RAID/toy/VR_Real/VR/"
        real = False

        self.object_image = None
        if not real:
            depth_cam_info_topic = "/unity/camera/left/depth/camera_info"
            depth_image_topic = "/unity/camera/left/depth/image_raw"
            rgb_cam_info_topic = "/unity/camera/left/rgb/camera_info"
            rgb_image_topic = "/unity/camera/left/rgb/image_raw"
        else:
            depth_cam_info_topic = "/left_camera/depth_registered/sw_registered/camera_info_throttled"
            depth_image_topic = "/left_camera/depth_registered/sw_registered/image_rect_throttled"
            rgb_cam_info_topic = "/left_camera/color/camera_info_throttled"
            rgb_image_topic = "/left_camera/color/image_raw_throttled"

        left_joint_topic = "/my_gen3_left/joint_states_throttled"
        right_joint_topic = "/my_gen3_right/joint_states_throttled"

        overlay_image_sub = rospy.Subscriber("/left_object_images", ObjectImage, self.get_image)
        depth_cam_info_pub = rospy.Publisher(depth_cam_info_topic, CameraInfo, queue_size=10)
        depth_image_pub = rospy.Publisher(depth_image_topic, Image, queue_size=10)
        rgb_cam_info_pub = rospy.Publisher(rgb_cam_info_topic, CameraInfo, queue_size=10)
        rgb_image_pub = rospy.Publisher(rgb_image_topic, Image, queue_size=10)
        left_states_pub = rospy.Publisher(left_joint_topic, JointState, queue_size=10)
        right_states_pub = rospy.Publisher(right_joint_topic, JointState, queue_size=10)

        data_path = "/home/rivr/text_image_data/"
        #text_file = open(os.path.join(data_path, "real_second_transcripts.txt"), "a")
        for file in os.listdir(path):
            bag_filename = os.path.join(path, os.fsdecode(file))
            print("=======================")
            print(bag_filename)
            a = file.split('_')
            participant = a[0]
            condition = a[1]
            os.makedirs(os.path.join(data_path, participant, condition), exist_ok=True)  # succeeds even if directory exists.
            bagfile = rosbag.Bag(bag_filename)
            for speech_topic, speech_msg, speech_t in bagfile.read_messages(topics="/transcript"):
                transcription = speech_msg.transcription
                audio_recieved = speech_msg.audio_recieved
                time = speech_t
                duration = speech_msg.duration
                print(f"{participant}, \t{condition}, \t{speech_t}, \t{transcription}")
                #text_file.write(f"{participant}, \t{condition}, \t{speech_t}, \t{transcription}\n")
                f = open(os.path.join(data_path, participant, condition, f"""{time}{"_".join(transcription.split(" "))}.txt"""), "a")
                f.write(transcription)
                f.close()
                
                '''
                depth_cam_info = self.get_last_msg(bagfile, speech_t, depth_cam_info_topic)
                depth_image = self.get_last_msg(bagfile, speech_t, depth_image_topic)
                rgb_cam_info = self.get_last_msg(bagfile, speech_t, rgb_cam_info_topic)
                rgb_image = self.get_last_msg(bagfile, speech_t, rgb_image_topic)
                left_states = self.get_last_msg(bagfile, speech_t, left_joint_topic)
                right_states = self.get_last_msg(bagfile, speech_t, right_joint_topic)

                if (depth_cam_info is not None) and (depth_image is not None) and (rgb_cam_info is not None) and (rgb_image is not None) and  (left_states is not None) and (right_states is not None):
                    for i in range(100):
                        t = rospy.Time.now()
                        depth_cam_info.header.stamp=t
                        depth_image.header.stamp=t
                        rgb_cam_info.header.stamp=t
                        rgb_image.header.stamp=t
                        rgb_cam_info.header.stamp=t
                        rgb_image.header.stamp=t
                        right_states.header.stamp=t
                        left_states.header.stamp=t
                        depth_cam_info_pub.publish(depth_cam_info)
                        depth_image_pub.publish(depth_image)
                        rgb_cam_info_pub.publish(rgb_cam_info)
                        rgb_image_pub.publish(rgb_image)
                        left_states_pub.publish(left_states)
                        right_states_pub.publish(right_states)
                    rospy.sleep(5)
                    rgb_img = self.bridge.imgmsg_to_cv2(rgb_image, desired_encoding="bgr8")
                    cv2.imwrite(os.path.join(data_path, participant, condition, f"{time}.png"), rgb_img)
                    if self.object_image is not None:
                        print(f"{self.object_image.header.stamp}")
                        overlay_img = self.bridge.imgmsg_to_cv2(self.object_image.image, desired_encoding="bgr8")
                        cv2.imwrite(os.path.join(data_path, participant, condition, f"{time}_overlay.png"), overlay_img)
                    else:
                        print('no objects')
                    self.object_image = None
                else:
                    print("no data")
                '''
        #text_file.close()

    def get_last_msg(self, bag, end_time, topic_name):
        start_time = end_time - rospy.Duration(3)
        last_msg = None
        last_t = None
        last_topic = None
        for topic, msg, t in bag.read_messages(topics=topic_name, start_time=start_time, end_time=end_time):
            last_topic = topic
            last_t = t
            last_msg = msg

        return last_msg

    def get_image(self, image):
        self.object_image = image


if __name__ == '__main__':
    data = ExtractData()

