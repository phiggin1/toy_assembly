#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from toy_assembly.msg import Transcription
from toy_assembly.srv import LLMImage
from toy_assembly.srv import LLMText
from toy_assembly.srv import TTS
from std_msgs.msg import Float32MultiArray

BLUE = (255, 0, 0)
GREEN = (0,255,0)
RED = (0,0,255)
PURPLE = (255,0,128)


def display_img(img):
    # show image
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
class TestLLM:
    def __init__(self):
        rospy.init_node('test_llm')
        cvbridge = CvBridge()

        self.robot_speech_pub = rospy.Publisher('/text_to_speech', String, queue_size=10)

        rospy.wait_for_service('llm_image')
        self.llm_img_serv = rospy.ServiceProxy('llm_image', LLMImage)

        rospy.wait_for_service('llm_text')
        self.llm_text_serv = rospy.ServiceProxy('llm_text', LLMText)

        rgb_image_topic =  "/unity/camera/left/rgb/image_raw"
        transcript_topic = "/transcript"

        rospy.loginfo(f"rgb_image_topic:{rgb_image_topic}")
        rospy.loginfo(f"transcript_topic:{transcript_topic}")
            
        rospy.loginfo("Wait for RGB image")
        rgb_image = rospy.wait_for_message(rgb_image_topic, Image) 
        rospy.loginfo("Got RGB image")

        #llm_img_serv(rgb_image)
        #rospy.loginfo(resp)
        
        text_sub = rospy.Subscriber("/transcript", Transcription, self.text_cb)
        rospy.spin()

    def text_cb(self, transcript):
        transcript =  transcript.transcription
        rospy.loginfo(f"Got transcript:'{transcript}'")

        resp = self.llm_text_serv(transcript)
        rospy.loginfo(resp)

        self.robot_speech_pub.publish(resp.text)


if __name__ == '__main__':
    test = TestLLM()

