#!/usr/bin/env python3
import sys
import rospy
import math 
import tf
import moveit_commander
import time
import json
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from toy_assembly.srv import Servo, ServoResponse
from toy_assembly.srv import LLMText, LLMTextRequest, LLMImageRequest
from toy_assembly.msg import Transcription
import numpy as np
from openai import OpenAI
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_inverse, quaternion_multiply


def quaternion_from_msg(orientation):
    return [orientation.x, orientation.y, orientation.z, orientation.w]

class Demo:
    def __init__(self):
        self.stop = False
        rospy.init_node('joining_exp')

        self.retry_times = 10      
        self.valid_target = False 

        
        self.robot_part_pub = rospy.Publisher("robot_text_topic", String, queue_size=10)
        self.human_part_pub = rospy.Publisher("human_text_topic", String, queue_size=10)

        
        '''
        key_filename = "/home/phiggin1/ai.key"
        with open(key_filename, "rb") as key_file:
            key = key_file.read().decode("utf-8")


        self.client = OpenAI(
            api_key = key,
        )
        '''

        self.tts_pub = rospy.Publisher('/text_to_speech', String, queue_size=10)

        
        self.finger_open = 1.0
        self.finger_partial_closed = 0.68
        self.finger_closed = 0.74
        self.hand_open = [self.finger_partial_closed, self.finger_partial_closed, self.finger_open]
        self.hand_closed = [self.finger_closed, self.finger_closed, self.finger_open]
        
        self.hand_over_pose = PoseStamped()
        self.hand_over_pose.header.frame_id = "base_link"
        self.hand_over_pose.pose.position.x =  0.50
        self.hand_over_pose.pose.position.y = -0.08
        self.hand_over_pose.pose.position.z =  0.95
        self.hand_over_pose.pose.orientation.x = -0.5
        self.hand_over_pose.pose.orientation.y = -0.5
        self.hand_over_pose.pose.orientation.z =  0.5
        self.hand_over_pose.pose.orientation.w =  0.5

        self.standoff_distance = 0.5    #m
        self.goal_tolerance = 0.0025    #m

        self.target = None
        self.last_time_spoke = rospy.Time.now().to_sec()
        self.presented = False
        
        joint_state_topic = ['joint_states:=/my_gen3_right/joint_states']

        moveit_commander.roscpp_initialize(joint_state_topic)

        self.robot = moveit_commander.RobotCommander(robot_description = "/my_gen3_right/robot_description")
        self.scene = moveit_commander.PlanningSceneInterface(ns="/my_gen3_right")

        rospy.loginfo("arm")
        self.arm_group_name = "arm"
        self.arm_move_group = moveit_commander.MoveGroupCommander(self.arm_group_name, ns="/my_gen3_right", robot_description = "/my_gen3_right/robot_description")
        self.arm_move_group.set_max_velocity_scaling_factor(0.25)
        self.arm_move_group.set_goal_position_tolerance(self.goal_tolerance)
        
        rospy.loginfo("gripper")
        self.hand_group_name = "gripper"
        self.hand_move_group = moveit_commander.MoveGroupCommander(self.hand_group_name, ns="/my_gen3_right", robot_description = "/my_gen3_right/robot_description")
        self.hand_move_group.set_max_velocity_scaling_factor(1.0)
        self.hand_move_group.set_goal_position_tolerance(self.goal_tolerance)
        

        rospy.wait_for_service('llm_text')
        self.llm_text_srv = rospy.ServiceProxy('llm_text', LLMText)


        self.planning_frame = "right_base_link"
        self.listener = tf.TransformListener()
        self.grab = rospy.Publisher('buttons', String, queue_size=10)
        self.target_sub = rospy.Subscriber("/human_slot_array", PoseArray, self.get_target)
        self.debug_publisher = rospy.Publisher("debug_standoff_pose", PoseStamped, queue_size=10)

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
        count = 0
        rate = rospy.Rate(1)
        while count < self.retry_times and not rospy.is_shutdown() and not self.valid_target:
            count += 1
            rate.sleep()
      
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

        np.set_printoptions(precision=3)

        print(f"\nq_ee            :{ q_ee}")
        print(f"q_ee            :{ (180/math.pi)*np.asarray( euler_from_quaternion(q_ee))}")
        print(f"q_robot_part    :{ q_robot_part}")
        print(f"q_robot_part    :{ (180/math.pi)*np.asarray(euler_from_quaternion(q_robot_part))}")
        print(f"q_target        :{ q_target}")
        print(f"q_target        :{ (180/math.pi)*np.asarray(euler_from_quaternion(q_target))}\n")

        '''
        # q_r to go from q_1 to q_2: 
        # q_r = q_2*q_1_inverse

        #this part is good (90 deg rot about y)
        #   quaternion_multiply(q_ee, q_robot_part_inverse) gives rot from part to ee
        
        q_r_part_ee = quaternion_multiply(q_ee, q_robot_part_inverse)
        print(f"q_r_part_ee     :{( q_r_part_ee)}")
        print(f"q_r_part_ee     :{(180/math.pi)*np.asarray(euler_from_quaternion( q_r_part_ee))}")
        

        #this part is good(identiy if lined up)
        #   quaternion_multiply(q_robot_part, q_target_inverse) give rot from target to part
        q_r_target_part = quaternion_multiply(q_robot_part, q_target_inverse)
        print(f"q_r_target_part :{( q_r_target_part)}")
        print(f"q_r_target_part :{(180/math.pi)*np.asarray(euler_from_quaternion( q_r_target_part))}\n")

        #quaternion_multiply(q_r_part_ee, q_r_target_part) give rot from target to ee
        q_r = quaternion_multiply(q_r_target_part, q_r_part_ee)


        q_r = quaternion_inverse(q_r_target_part)
        q_r = q_r_target_part
        print(f"q_r             :{( q_r)}")
        print(f"q_r             :{(180/math.pi)*np.asarray(euler_from_quaternion( q_r))}")


        q_new = quaternion_multiply(q_r,q_ee)
        print(f"q_new           :{( q_new)}")
        print(f"q_new           :{(180/math.pi)*np.asarray(euler_from_quaternion( q_new))}")

        #apply q_r but transform into eef frame

        #q_true = quaternion_from_euler(0, 180*(math.pi/180), 135*(math.pi/180))
        #print(f"q_true           :{( q_true)}")
        #print(f"q_true           :{(180/math.pi)*np.asarray(euler_from_quaternion( q_true))}")


        #self.target.pose.position.z += .1
        '''
        q_new = q_target


        self.target.pose.orientation.x = q_new[0]
        self.target.pose.orientation.y = q_new[1]
        self.target.pose.orientation.z = q_new[2]
        self.target.pose.orientation.w = q_new[3]

        self.debug_publisher.publish(self.target)

        return self.target


    #add checking if sucess
    def move_arm(self, pose, speed):
        status = True
        
        self.arm_move_group.set_max_velocity_scaling_factor(speed)
        self.arm_move_group.set_pose_target(pose.pose)
        status = self.arm_move_group.go(wait=True)
        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()
        
        return status

    def move_fingers(self, finger_positions):
        self.hand_move_group.go(finger_positions, wait=True)

    def servo(self):
        rospy.wait_for_service('Servo')
        try:
            joining_servo = rospy.ServiceProxy('Servo', Servo)
            resp = joining_servo()
            return resp.resp
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def get_gpt_response(self, statement):
        req = LLMTextRequest()
        req.text = statement
        resp = self.llm_text_srv(req)

        text = resp.text

        a = text.find('{')
        b = text.find('}')+1
        text_json = text[a:b]
        json_dict = json.loads(text_json)

        return json_dict

    def experiment(self):
        
        
        p = self.arm_move_group.get_current_pose()
        #rospy.loginfo(f"Inital ee orientation :\n{p.pose.orientation}\n")
        
            #re home the arm
        
        self.arm_move_group.set_max_velocity_scaling_factor(1.0)
        self.arm_move_group.set_named_target('home')
        self.arm_move_group.go(wait=True)
        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()
        
        
        robot_asks = "what objects are you going to pick up, and what object should I pick up?"
        self.tts_pub.publish(robot_asks)
        #call tts service

        #human_reply = rospy.wait_for_message("/transcript", Transcription)
        human_reply = "Can you pick up the red legs, I am going to pickup the blue body."
        #wait for transcript msg
        rospy.loginfo(human_reply)

        #querty GPT for response
        resp = self.get_gpt_response(human_reply)
        h = String()
        h.data = resp["human"][1:-1]
        r = String()
        r.data = resp["robot"][1:-1]
        print(f"human: {h}")
        print(f"robot: {r}")

        #publish what object the pose tracking componest should look for
        rate = rospy.Rate(50)
        for i in range(10):
            self.robot_part_pub.publish(r)
            self.human_part_pub.publish(h)
            rate.sleep()
        
        #get the pose of a slot of the target object
        standoff_pose = self.get_init_target()

        #move the arm above it
        standoff_pose.pose.position.z += self.standoff_distance
        #rospy.loginfo(f"standoff pose :\n{standoff_pose}\n")
        
        success = False
        success = self.move_arm(standoff_pose, 1.0)
        attempts = 1
        print(f"{success}, attempt:{attempts}, {standoff_pose.pose.position.z}")

        #if the robot cannot find a plan update the standoff pose
        # a little hight above
        
        while not success and attempts <= 5 and standoff_pose.pose.position.z > 0.0:
            #rospy.loginfo('moving to new standoff')
            standoff_pose.pose.position.z -= 0.1
            success = self.move_arm(standoff_pose, 1.0)
            attempts += 1
            print(f"{success}, attempt:{attempts}, {standoff_pose.pose.position.z}")
        
        #print(f"{success}, attempt:{attempts}, {standoff_pose.pose}")

        
        if success:
            #try and servoce the part into position
            rospy.loginfo('pre servo')
            print(self.servo())
            rospy.loginfo('post servo')

            rospy.loginfo('moving to standoff')
            success = self.move_arm(standoff_pose, 1.0)
            print(f"{success}, {standoff_pose.pose}")


        
        



if __name__ == '__main__':
    demo = Demo()
    demo.experiment()
