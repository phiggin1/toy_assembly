#!/usr/bin/env python3

import rospy
import json
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from toy_assembly.srv import LLMText, LLMTextRequest
from toy_assembly.srv import LLMImage, LLMImageRequest
from toy_assembly.srv import TTS, TTSRequest, TTSResponse
from toy_assembly.srv import Servo
from toy_assembly.srv import MoveITPose, MoveITPoseRequest
from toy_assembly.srv import MoveITGrabPose, MoveITGrabPoseRequest
from toy_assembly.msg import Transcription
from copy import deepcopy

class Demo:
    def __init__(self):
        self.stop = False
        rospy.init_node('toy_assembly_demo')

        self.standoff_distance = 0.3

        self.robot_part_pub = rospy.Publisher("robot_text_topic", String, queue_size=10)
        self.human_part_pub = rospy.Publisher("human_text_topic", String, queue_size=10)
        self.rivr_robot_speech = rospy.Publisher('/robotspeech', Float32MultiArray, queue_size=10)


        rospy.wait_for_service('get_text_to_speech')
        self.tts_serv = rospy.ServiceProxy('get_text_to_speech', TTS)


        rospy.wait_for_service('llm_text')
        self.llm_text_srv = rospy.ServiceProxy('llm_text', LLMText)


    def get_init_robot_target(self):
        robot_part_pose = rospy.wait_for_message("/robot_slot_array", PoseArray)

        target = PoseStamped()
        target.header = robot_part_pose.header
        target.pose = robot_part_pose.poses[0]
        t = rospy.Time.now()
        target.header.stamp = t

        rospy.loginfo("robot target pose:")
        print(target)

        return target


    def get_init_human_target(self):
        human_part_pose = rospy.wait_for_message("/human_slot_array", PoseArray)

        target = PoseStamped()
        target.header = human_part_pose.header
        target.pose = human_part_pose.poses[0]
        t = rospy.Time.now()
        target.header.stamp = t

        rospy.loginfo("human target pose:")
        print(target)

        return target

    def get_gpt_response(self, statement):
        req = LLMTextRequest()
        req.text = statement
        resp = self.llm_text_srv(req)
        text = resp.text
       
        rospy.loginfo(f"gpt reponse:{text}")
        rospy.loginfo("===================================")

        #extract the dictonary
        a = text.find('{')
        b = text.find('}')+1
        text_json = text[a:b]
        json_dict = json.loads(text_json)

        #return just the dict
        return json_dict

    def servo(self):
        rospy.wait_for_service('Servo')
        try:
            servo = rospy.ServiceProxy('Servo', Servo)
            resp = servo()
            return resp.resp
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s"%e)

    def right_arm_move_to_pose(self, pose):
        rospy.wait_for_service('/my_gen3_right/move_pose')
        print("right_arm_move_to_pose")
        try:
            moveit_pose = rospy.ServiceProxy('/my_gen3_right/move_pose', MoveITPose)
            resp = moveit_pose(pose)
            return resp
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s"%e)

    def right_arm_grab(self, pose):
        rospy.wait_for_service('/my_gen3_right/grab_object')
        print("right_arm_grab")
        try:
            moveit_pose = rospy.ServiceProxy('/my_gen3_right/grab_object', MoveITGrabPose)
            resp = moveit_pose(pose)
            return resp
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s"%e)

    def experiment(self):        
        
        rospy.loginfo("===================================")
        robot_asks = "What objects are you going to pick up, and what object should the robot pick up?"
        req = TTSRequest()
        req.text = robot_asks
        resp = self.tts_serv(req)
        float_audio_array = resp.audio
        self.rivr_robot_speech.publish(float_audio_array)
        rospy.loginfo(f"robot asks:{robot_asks}")
        rospy.loginfo("===================================")
        
        '''
        overlayed_image = rospy.wait_for_message("/overlayed_images", Image)
        gaze_targets = rospy.wait_for_message("/gaze_targets", Float32MultiArray)
        rospy.loginfo(gaze_targets)
        '''

        human_reply = rospy.wait_for_message("/transcript", Transcription)
        human = human_reply.transcription
        
        #human = "Can you pick up the yellow body, I am going to pickup the red legs."
        #human = "I was gonna pick up red lace, can pick up blue bob."
        #human = "I'm going to pick up the red links, pick up blue by the"

        
        rospy.loginfo(f"human:{human}")
        rospy.loginfo(f"===================================")

        
        #querty GPT for response
        resp = self.get_gpt_response(human)
        rospy.loginfo(resp)
        '''
        h = String()
        h.data = "red_horse_front_legs"
        r = String()
        r.data = "horse_body_yellow"
        '''

        h = String()
        h.data = resp["human"][1:-1]
        print(h.data)
        r = String()
        r.data = resp["robot"][1:-1]
        print(r.data)
        rospy.loginfo(f"===================================")



        #publish what object the pose tracking componest should look for
        # redo this as a service possib;y
        rate = rospy.Rate(50)
        for i in range(5):
            self.robot_part_pub.publish(r)
            self.human_part_pub.publish(h)
            rate.sleep()

        #get target location of robot part
        robot_part_pose = self.get_init_robot_target()
        
        rospy.loginfo("tell robot to grab robot part")
        print(r)
        status = self.right_arm_grab(robot_part_pose)
        print(status)

        rospy.loginfo(f"===================================")
        #a = input("waiting...")

        rospy.loginfo("get target location of human part")
        print(h)
        human_part_pose = self.get_init_human_target()
        standoff_pose = deepcopy(human_part_pose)
        standoff_pose.pose.position.z += self.standoff_distance
        print(f"standoff\n{standoff_pose.pose.position}")
        #move robot part above human part
        status = self.right_arm_move_to_pose(standoff_pose)
        rospy.loginfo(status)
        while not status and standoff_pose.pose.position.z > 0.1:
            standoff_pose.pose.position.z -= 0.1
            print(f"standoff\n{standoff_pose.pose.position.z}")
            status = self.right_arm_move_to_pose(standoff_pose)
            rospy.loginfo(status)

        print(status)
        rospy.loginfo(f"===================================")
        #a = input("waiting...")
        
        rospy.loginfo("servo robot part into human part")
        status = self.servo()
        print(status)
        
        
if __name__ == '__main__':
    demo = Demo()
    demo.experiment()
