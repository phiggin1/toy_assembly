#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
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

rospy.init_node('test_llm')
cvbridge = CvBridge()


robot_speech_pub = rospy.Publisher('/text_to_speech', String, queue_size=10)

rospy.wait_for_service('llm_image')
llm_img_serv = rospy.ServiceProxy('llm_image', LLMImage)

rospy.wait_for_service('llm_text')
llm_text_srv = rospy.ServiceProxy('llm_text', LLMText)


rgb_image_topic =  "/unity/camera/left/rgb/image_raw"
transcript_topic = "/transcript"

rospy.loginfo(f"rgb_image_topic:{rgb_image_topic}")
rospy.loginfo(f"transcript_topic:{transcript_topic}")
            
rgb_image = rospy.wait_for_message(rgb_image_topic, Image) 
rospy.loginfo("Got RGB image")

resp = llm_img_serv(rgb_image)
rospy.loginfo(resp)

transcript = rospy.wait_for_message(transcript_topic, Transcription)
rospy.loginfo("Got transcript")

transcript =  [transcript.transcription]
rospy.loginfo(transcript) 

resp = llm_text_serv(transcript)
rospy.loginfo(resp)

robot_speech_pub.publish(resp.text)