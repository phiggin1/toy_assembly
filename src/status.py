#!/usr/bin/env python3

import json
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from toy_assembly.msg import Transcription
import numpy as np
from cv_bridge import CvBridge
import cv2
import textwrap 

class Status:
    def __init__(self):
        rospy.init_node('Status', anonymous=True)
        self.real = rospy.get_param("~real", default=False)
        
        self.cv_bridge = CvBridge()

        self.robot_text = None
        self.last_robot_text = None
        self.human_text = None
        self.last_human_text = None
        self.status = "WAITING"

        self.display_pub = rospy.Publisher("/display", String, queue_size=10)

        self.sub = rospy.Subscriber("status", String, self.status_cb)
        self.transcript_sub = rospy.Subscriber("transcript", Transcription, self.transcript_cb)
        self.tts_sub = rospy.Subscriber("text_to_speech", String, self.robot_speech_cb)

        self.rate = rospy.Rate(5) # 5hz
        while not rospy.is_shutdown():
            msg_text = {
                "status":self.status
            }

            if self.last_robot_text is not None:
                if self.last_robot_text+rospy.Duration(5) > rospy.Time.now():
                    msg_text["robot"] = self.robot_text
            else:
                msg_text["robot"] = ""

            if self.last_human_text is not None:
                if self.last_human_text+rospy.Duration(5) > rospy.Time.now():
                    msg_text["human"] = self.human_text
                else:
                    msg_text["human"] = ""
            else:
                msg_text["human"] = ""

            #rospy.loginfo(msg_text)

            if self.real:
                self.display_status(msg_text["status"] , msg_text["human"])

            msg = String()
            msg.data = json.dumps(msg_text)
            self.display_pub.publish(msg)

            self.rate.sleep()


    def display_status(self, status, human_text):
        #BGR color
        if status == "THINKING":
            #red
            print("red")
            color = (0, 0, 255)
        elif status == "LISTENING":
            #yellow
            print("yellow")
            color = (0, 196, 196)
        else:
            print("green")
            #green
            color = (0, 255, 0)
            
        thickness = 2

        status_img = np.zeros((480, 640, 3))
        start_point = (5,5)
        end_point = (635, 65)
        cv2.rectangle(status_img, start_point, end_point, color, -1)
        
        font = cv2.FONT_HERSHEY_DUPLEX 
        org = (50, 50)
        fontScale = 1.5
        cv2.putText(status_img, status, org, font, fontScale, (255,255,255), thickness, cv2.LINE_AA)
        
        wrapped_text = textwrap.wrap(human_text, width=36)
        for (i, line) in enumerate(wrapped_text):
            textsize = cv2.getTextSize(line, font, fontScale, thickness)[0]
            gap = textsize[1] + 5
            y= int((200 + textsize[1]) / 2) + i * gap
            cv2.putText(status_img, line, (5, y) , font, fontScale, (255,255,255), thickness, cv2.LINE_AA)
        
        cv2.imshow("status", status_img)   
        cv2.waitKey(delay = 1)


    def transcript_cb(self, msg):
        self.human_text = msg.transcription
        self.last_human_text = rospy.Time.now()


    def robot_speech_cb(self, msg):
        self.robot_text = msg
        self.last_robot_text = rospy.Time.now()

    def status_cb(self, msg):
        status_msg = msg.data
        if self.status == "WAITING":
            if status_msg == "LISTENING":
                self.status = "LISTENING"
                #rospy.loginfo("WAITING > LISTENING")
        elif self.status == "LISTENING":
            if status_msg == "THINKING":
                self.status = "THINKING"
                #rospy.loginfo("LISTENING > THINKING")
        elif self.status == "THINKING":
            if status_msg == "WAITING":
                self.status = "WAITING"
                #rospy.loginfo("THINKING > WAITING")

if __name__ == '__main__':
    status = Status()

