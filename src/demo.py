#!/usr/bin/env python3
import sys
import rospy
import math 
import tf
import moveit_commander
import time
import json

from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from toy_assembly.srv import Servo, ServoRequest, ServoResponse
from toy_assembly.srv import MoveITPose, MoveITPoseRequest, MoveITPoseResponse
from toy_assembly.srv import MoveITNamedPose, MoveITNamedPoseRequest, MoveITNamedPoseResponse

from toy_assembly.srv import LLMText, LLMTextRequest, LLMImageRequest
from toy_assembly.srv import TTS, TTSRequest, TTSResponse
from toy_assembly.msg import Transcription
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_inverse, quaternion_multiply


def quaternion_from_msg(orientation):
    return [orientation.x, orientation.y, orientation.z, orientation.w]

class Demo:
    def __init__(self):
        self.stop = False
        rospy.init_node('toy_assembly')

        self.retry_times = 10      
        self.valid_target = False 

        self.target = None
        self.last_time_spoke = rospy.Time.now().to_sec()
        self.presented = False

        self.listener = tf.TransformListener()
        self.grab = rospy.Publisher("/buttons", String, queue_size=10)
        self.debug_publisher = rospy.Publisher("debug_standoff_pose", PoseStamped, queue_size=10)
        self.robot_part_pub = rospy.Publisher("robot_text_topic", String, queue_size=10)
        self.human_part_pub = rospy.Publisher("human_text_topic", String, queue_size=10)


        rospy.wait_for_service('get_text_to_speech')
        self.tts_serv = rospy.ServiceProxy('get_text_to_speech', TTS)

        self.rivr_robot_speech = rospy.Publisher('/robotspeech', Float32MultiArray, queue_size=10)

        rospy.wait_for_service('llm_text')
        self.llm_text_srv = rospy.ServiceProxy('llm_text', LLMText)

        self.target_sub = rospy.Subscriber("/human_slot_array", PoseArray, self.get_target)


    def get_target(self, human_part_pose):
        #print(human_part_pose.header.frame_id)
        #print(self.planning_frame)
        target = PoseStamped()
        target.header = human_part_pose.header
        target.pose = human_part_pose.poses[0]
        t = rospy.Time.now()
        target.header.stamp = t

        #print(target.pose.orientation)
        self.listener.waitForTransform(target.header.frame_id, self.planning_frame, t, rospy.Duration(4.0) )
        self.target = self.listener.transformPose(self.planning_frame, target)  
        #print(target.pose.orientation)

        self.last_valid_target = rospy.Time.now()  
        self.valid_target = True

    def get_init_target(self):
        human_part_pose = rospy.wait_for_message("/human_slot_array", PoseArray)
        self.target = PoseStamped()
        self.target.header = human_part_pose.header
        self.target.pose = human_part_pose.poses[0]
        t = rospy.Time.now()
        self.target.header.stamp = t
        print(self.target)

        #print(target.pose.orientation)
        self.listener.waitForTransform(self.target.header.frame_id, self.planning_frame, t, rospy.Duration(4.0) )
        self.target = self.listener.transformPose(self.planning_frame, self.target)  


        '''count = 0
        rate = rospy.Rate(1)
        while count < self.retry_times and not rospy.is_shutdown() and not self.valid_target:
            count += 1
            rate.sleep()'''
      
        robot_part_pose = rospy.wait_for_message("/robot_slot_array", PoseArray)
        robot_part_pose = robot_part_pose.poses[0]
        q_robot_part = np.asarray([
            robot_part_pose.orientation.x,
            robot_part_pose.orientation.y,
            robot_part_pose.orientation.z,
            robot_part_pose.orientation.w

        ])

        ee_pose = self.arm_move_group.get_current_pose()
        q_ee = np.asarray([
            ee_pose.pose.orientation.x,
            ee_pose.pose.orientation.y,
            ee_pose.pose.orientation.z,
            ee_pose.pose.orientation.w
        ])


        q_target = np.asarray([
            self.target.pose.orientation.x,
            self.target.pose.orientation.y,
            self.target.pose.orientation.z,
            self.target.pose.orientation.w
        ])
        
        q_ee_inverse = quaternion_inverse(q_ee)
        q_target_inverse = quaternion_inverse(q_target) 
        q_robot_part_inverse = quaternion_inverse(q_robot_part)

        '''
        np.set_printoptions(precision=3)
        print(f"\nq_ee            :{ q_ee}")
        print(f"q_ee            :{ (180/math.pi)*np.asarray( euler_from_quaternion(q_ee))}")
        print(f"q_robot_part    :{ q_robot_part}")
        print(f"q_robot_part    :{ (180/math.pi)*np.asarray(euler_from_quaternion(q_robot_part))}")
        print(f"q_target        :{ q_target}")
        print(f"q_target        :{ (180/math.pi)*np.asarray(euler_from_quaternion(q_target))}\n")
        '''
        q_new = q_target

        self.target.pose.orientation.x = q_new[0]
        self.target.pose.orientation.y = q_new[1]
        self.target.pose.orientation.z = q_new[2]
        self.target.pose.orientation.w = q_new[3]

        self.debug_publisher.publish(self.target)

        return self.target




    def get_gpt_response(self, statement):
        rospy.loginfo('----------------------')
        rospy.loginfo(statement)
        req = LLMTextRequest()
        req.text = statement
        resp = self.llm_text_srv(req)
        text = resp.text


        '''
        text = """{
"robot": "<horse_body_blue>",
"human": "<red_horse_front_legs>",
"question": ""
}"""
        '''

        rospy.loginfo(text)
        a = text.find('{')
        b = text.find('}')+1
        text_json = text[a:b]
        json_dict = json.loads(text_json)

        return json_dict

    def experiment(self):


        
        robot_asks = "what objects are you going to pick up, and what object should I pick up?"
        req = TTSRequest()
        req.text = robot_asks
        resp = self.tts_serv(req)
        float_audio_array = resp.audio
        self.rivr_robot_speech.publish(float_audio_array)
        
        #call tts service
        '''
        "robot": "<horse_body_yellow>",
        "human": "<red_horse_front_legs>"
        '''
        
        '''
        overlayed_image = rospy.wait_for_message("/overlayed_images", Image)
        gaze_targets = rospy.wait_for_message("/gaze_targets", Float32MultiArray)
        rospy.loginfo(gaze_targets)
        '''

        #human_reply = rospy.wait_for_message("/transcript", Transcription)
        #human = human_reply.transcription
        #human = "Can you pick up the yellow body, I am going to pickup the red legs."
        #human = "I was gonna pick up red lace, can pick up blue bob."
        #human = "I'm going to pick up the red links, pick up blue by the"
        human = "Can you pick the blue body or is it going to pick the red eggs?"
        rospy.loginfo(human)

        #querty GPT for response
        resp = self.get_gpt_response(human)

        print(resp)
        h = String()
        h.data = resp["human"][1:-1]
        r = String()
        r.data = resp["robot"][1:-1]
        print(f"human: {h}")
        print(f"robot: {r}")

        #publish what object the pose tracking componest should look for
        rate = rospy.Rate(50)
        for i in range(5):
            self.robot_part_pub.publish(r)
            self.human_part_pub.publish(h)
            rate.sleep()



        
if __name__ == '__main__':
    demo = Demo()
    demo.experiment()
