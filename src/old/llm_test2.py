#!/usr/bin/env python3

import zmq
import numpy as np
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import String
from toy_assembly.msg import Transcription
from toy_assembly.srv import TTS, TTSResponse
from multiprocessing import Lock
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_srvs.srv import Trigger 
import json

class LLMClient:
    def __init__(self):
        rospy.init_node('LLM_testing')

        self.debug = rospy.get_param("~debug", True)
        server_port = rospy.get_param("~port", "8877")

        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % server_port)
        rospy.loginfo(f"Server listening on port:{server_port}")

        self.robot_speech_pub = rospy.Publisher('/text_to_speech', String, queue_size=10)
      
        self.twist_topic  = "/my_gen3_right/servo/delta_twist_cmds"
        self.cart_vel_pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=10)
        rospy.loginfo(self.twist_topic)


        '''
        self.text_sub = rospy.Subscriber("/transcript", Transcription, self.text_cb)
        rospy.spin()
        '''
        while True:
            t = Transcription()
            text = input()
            t.transcription = text
            self.text_cb(t)
        
    def send_ada(self, text):

        msg = {"type":"llm",
               "text":text
        }

        if self.debug: rospy.loginfo("============================") 
        if self.debug: print(f"LLM sending to ada\ntext:{text}")
        
        self.socket.send_json(msg)
        resp = self.socket.recv_json()

        if self.debug: rospy.loginfo('LLM  recv from ada') 

        if "error" in resp:
            print(resp["error"])
            return

        rospy.loginfo(resp)
        text = resp["text"]
        #extract the dictonary
        a = text.find('{')
        b = text.find('}')+1
        text_json = text[a:b]
        json_dict = json.loads(text_json)

        action = None
        if "action" in json_dict:
            action = json_dict["action"]

        return action

    def text_cb(self, transcript):
        transcript =  transcript.transcription

        action = self.send_ada(transcript)

        print(f"action: - {action} - ")

        if "PICKUP" in action:
            self.send_gpt(transcript)
        else:
            self.ee_move(action)
            
    def send_gpt(self, transcript):
        rospy.loginfo(f"pickup transcript:{transcript}")


    def ee_move(self, action):
        speed = 0.1
        angular_speed = 1.0

        #Servo in EE base_link frame
        move = False
        x = 0.0
        y = 0.0
        z = 0.0
        roll = 0.0
        yaw = 0.0
        pitch = 0.0

        if "PITCH_UP" in action:
            rospy.loginfo("PITCH_UP")
            pitch =-angular_speed
            move = True
        elif "PITCH_DOWN" in action:
            rospy.loginfo("PITCH_DOWN")
            pitch = angular_speed
            move = True

        if  "ROTATE_LEFT" in action:
            rospy.loginfo("ROLL_LEFT")
            roll =-angular_speed
            move = True
        elif "ROTATE_RIGHT" in action:
            rospy.loginfo("ROLL_RIGHT")
            roll = angular_speed
            move = True
        
        if "YAW_LEFT" in action:
            rospy.loginfo("YAW_LEFT")
            yaw =-angular_speed
            move = True
        elif "YAW_RIGHT" in action:
            rospy.loginfo("YAW_RIGHT")
            yaw = angular_speed
            move = True
    
        if "MOVE_FORWARD" in action:
            rospy.loginfo("MOVE_FORWARD")
            x = speed
            move = True
        elif "MOVE_BACKWARD" in action:
            rospy.loginfo("MOVE_BACKWARD")
            x =-speed
            move = True

        if "MOVE_RIGHT" in action:
            rospy.loginfo("MOVE_RIGHT")
            y =-speed
            move = True
        elif "MOVE_LEFT" in action:
            rospy.loginfo("MOVE_LEFT")
            y = speed
            move = True
        
        if "MOVE_UP" in action:
            rospy.loginfo("MOVE_UP")
            z = speed
            move = True
        elif "MOVE_DOWN" in action:
            rospy.loginfo("MOVE_DOWN")
            z =-speed
            move = True
        
        if move:
            self.move(x,y,z, roll, pitch, yaw)

        if "CLOSE_HAND" in action:
            self.grab()
        if "OPEN_HAND" in action:
            self.release()            

    def move(self, x, y, z, roll, pitch, yaw):
        cmd = TwistStamped()
        #cmd.header.frame_id ="right_end_effector_link"
        cmd.header.frame_id ="right_base_link"
        cmd.twist.linear.x = x
        cmd.twist.linear.y = y
        cmd.twist.linear.z = z
        cmd.twist.angular.x = roll
        cmd.twist.angular.y = pitch
        cmd.twist.angular.z = yaw

        print(cmd.header.frame_id)
        print(x,y,z,roll,pitch,yaw)

        num_msgs = 50
        rate = rospy.Rate(100)
        for i in range(num_msgs):
            cmd.header.stamp = rospy.Time.now()
            self.cart_vel_pub.publish(cmd)
            rate.sleep()
        for i in range(10):
            cmd.twist.linear.x = 0
            cmd.twist.linear.y = 0
            cmd.twist.linear.z = 0
            cmd.twist.angular.x = 0
            cmd.twist.angular.y = 0
            cmd.twist.angular.z = 0
            cmd.header.stamp = rospy.Time.now()
            self.cart_vel_pub.publish(cmd)
            rate.sleep()

    def grab(self):
        service_name = "/my_gen3_right/close_hand"
        rospy.wait_for_service(service_name)
        print(service_name)
        try:
            close_hand = rospy.ServiceProxy(service_name, Trigger)
            resp = close_hand()
            return resp
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s"%e)
    
    def release(self):
        service_name = "/my_gen3_right/open_hand"
        rospy.wait_for_service(service_name)
        print(service_name)
        try:
            open_hand = rospy.ServiceProxy(service_name, Trigger)
            resp = open_hand()
            return resp
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s"%e)

if __name__ == '__main__':
    llm = LLMClient()
