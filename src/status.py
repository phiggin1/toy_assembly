#!/usr/bin/env python3

import json
import rospy
from std_msgs.msg import String
from toy_assembly.msg import Transcription
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Status:
    def __init__(self):
        rospy.init_node('Status', anonymous=True)

        self.display_pub = rospy.Publisher("/display", String, queue_size=10)

        self.sub = rospy.Subscriber("status", String, self.status_cb)
        self.transcript_sub = rospy.Subscriber("transcript", Transcription, self.transcript_cb)
        self.tts_sub = rospy.Subscriber("text_to_speech", String, self.robot_speech_cb)

        self.robot_text = None
        self.last_robot_text = None
        self.human_text = None
        self.last_human_text = None
        self.status = "WAITING"

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

            #BGR color
            if self.status == "LISTENING":
                #yellow
                color = (33, 222, 255)
            elif self.status == "THINKING":
                #red
                color = (0, 0, 255)
            else:
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
            cv2.putText(status_img, self.status, org, font, fontScale, (255,255,255), thickness, cv2.LINE_AA)
  
            b,g,r = cv2.split(status_img)
            frame_rgb = cv2.merge((r,g,b))
            plt.imshow(frame_rgb)
            plt.axis('off')            
            plt.show()

            msg = String()
            msg.data = json.dumps(msg_text)
            self.display_pub.publish(msg)
            #rospy.loginfo(msg_text)

            self.rate.sleep()

    def transcript_cb(self, msg):
        self.human_text = msg.transcription
        self.last_human_text = rospy.Time.now()

    def robot_speech_cb(self, msg):
        self.robot_text = msg
        self.last_robot_text = rospy.Time.now()

    def status_cb(self, msg):
        status_msg = msg.data

        rospy.loginfo(status_msg)

        if self.status == "WAITING":
            if status_msg == "LISTENING":
                self.status = "LISTENING"
                rospy.loginfo("WAITING > LISTENING")
        elif self.status == "LISTENING":
            if status_msg == "THINKING":
                self.status = "THINKING"
                rospy.loginfo("LISTENING > THINKING")
        elif self.status == "THINKING":
            if status_msg == "WAITING":
                self.status = "WAITING"
                rospy.loginfo("THINKING > WAITING")

if __name__ == '__main__':
    status = Status()

