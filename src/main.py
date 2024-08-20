#!/usr/bin/env python3

import zmq
import json
from copy import deepcopy
import rospy
import tf
from cv_bridge import CvBridge
from toy_assembly.msg import Transcription, ObjectImage
from toy_assembly.srv import LLMImage, LLMImageRequest
from toy_assembly.srv import MoveITPose
from geometry_msgs.msg import TwistStamped, PoseStamped, PointStamped
from std_srvs.srv import Trigger 
from std_msgs.msg import String

def extract_json(text):
    a = text.find('{')
    b = text.find('}')+1
    text_json = text[a:b]
    json_dict = json.loads(text_json)

    return json_dict

class AssemblyClient:
    def __init__(self):
        rospy.init_node('toy_assembly_main')
        self.cvbridge = CvBridge()
        self.listener = tf.TransformListener()

        self.real = rospy.get_param("~real", True)

        self.num_msgs = 20
        self.rate = rospy.Rate(20)
        self.speed = 0.1
        self.angular_speed = 1.0

        self.state = "LOW_LEVEL"
        self.check = False
        self.debug = rospy.get_param("~debug", True)
        server_port = rospy.get_param("~port", "8877")

        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % server_port)
        rospy.loginfo(f"Server listening on port:{server_port}")

        self.llm_text_srv = rospy.ServiceProxy('gpt_servcice', LLMImage)
        self.llm_check_srv = rospy.ServiceProxy('gpt_servcice_check', LLMImage)
        self.robot_speech_pub = rospy.Publisher('/text_to_speech', String, queue_size=10)
      
        self.debug_pose_pub = rospy.Publisher('/0debug_pose', PoseStamped, queue_size=10)

        self.status_pub = rospy.Publisher("/status", String, queue_size=10)

        self.twist_topic  = "/my_gen3_right/workspace/delta_twist_cmds"
        self.cart_vel_pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=10)
        rospy.loginfo(self.twist_topic)

        #self.text_sub = rospy.Subscriber("/transcript", Transcription, self.text_cb)
        #rospy.spin()
        
        while not rospy.is_shutdown():
            transcription = rospy.wait_for_message("/transcript", Transcription)
            self.text_cb(transcription)

    def text_cb(self, transcript):
        rospy.loginfo("THINKING")
        self.status_pub.publish("THINKING")

        if self.debug: rospy.loginfo("========================================================") 
        if self.debug: rospy.loginfo(f"{transcript}")
        if self.debug: rospy.loginfo(f"state: {self.state}")

        transcript =  transcript.transcription
        if self.state == "HIGH_LEVEL":
            self.high_level(transcript)
        else:
            self.low_level(transcript)
    
        rospy.loginfo("WAITING")
        self.status_pub.publish("WAITING")
        if self.debug: rospy.loginfo("--------------------------------------------------------") 

    def high_level(self, text):
        rospy.loginfo("sending to gpt")

        overlay_topic = "/left_object_images"
        obj_img = rospy.wait_for_message(overlay_topic, ObjectImage)
        frame = obj_img.header.frame_id
        image = obj_img.image
        opject_positions = obj_img.object_positions
        opject_positions = self.transform_positions(opject_positions, frame, "world")

        objects = []
        for i in range(len(opject_positions)):
            objects.append(f"obj_{i}")

        for i in range(len(opject_positions)):
            rospy.loginfo(f"{objects[i]}, {opject_positions[i].header.frame_id}, {opject_positions[i].point.x}, {opject_positions[i].point.y},{opject_positions[i].point.z}")

        req = LLMImageRequest()
        req.text = text
        req.objects = objects
        req.check = self.check
        req.image = image

        rospy.loginfo(f"init: {req.text}, {req.objects}, {req.check}")

        resp = self.llm_text_srv(req)
        #Should do some error checking
        # in future
        json_dict = extract_json(resp.text)
        rospy.loginfo(f"init\n{json_dict}")
        obj = json_dict["object"]
        indx = objects.index(obj)
        self.target_position = opject_positions[indx]
        print(self.target_position)
        self.indicate(self.target_position)
        self.grab(self.target_position)
        #Reset the state
        self.state = "LOW_LEVEL"

        '''
        correct = False
        if not self.check:
            rospy.loginfo(f"init: {req.text}, {req.objects}, {req.check}")
            resp = self.llm_text_srv(req)
            #Should do some error checking
            # in future
            json_dict = extract_json(resp.text)
            rospy.loginfo(f"init\n{json_dict}")
            obj = json_dict["object"]
            indx = objects.index(obj)
            self.target_position = opject_positions[indx]
            print(self.target_position)
        else:
            rospy.loginfo(f"check: {req.text}, {req.objects}, {req.check}")
            resp = self.llm_check_srv(req)
            #Should do some error checking
            # in future
            json_dict = extract_json(resp.text)
            rospy.loginfo(f"check\n{json_dict}")
            if json_dict["correct"] == "True":
                correct = True
    
        rospy.loginfo(correct)
        rospy.loginfo("------")

        if correct:
            print(self.target_position)
            self.grab(self.target_position)
            #Reset the state
            self.check = False
            self.state = "LOW_LEVEL"
        else:
            obj = json_dict["object"]
            indx = objects.index(obj)
            self.target_position = opject_positions[indx]
            self.indicate(self.target_position)
            self.robot_speech_pub.publish("This one?")
            self.check = True
        '''

    def low_level(self, text):
        action = self.send_ada(text)
        rospy.loginfo(f"low level action:\n\t {action}")

        if action is None:
            return

        if "PICKUP" in action:
            self.state = "HIGH_LEVEL"
            self.high_level(text)            
            rospy.loginfo("TESTING")
        else:   
            rospy.loginfo(f"state: {self.state}")
            self.state = "LOW_LEVEL"
            self.ee_move(action)

    def send_ada(self, text):
        msg = {"type":"llm",
               "text":text
        }

        if self.debug: rospy.loginfo(f"LLM sending to ada\ntext:{text}")
        
        self.socket.send_json(msg)
        resp = self.socket.recv_json()

        if self.debug: rospy.loginfo('LLM  recv from ada') 

        if "error" in resp:
            rospy.loginfo(resp["error"])
            return None

        text = resp["text"]
        rospy.loginfo(f"response:\n{text}\n ---")
        
        try:
            json_dict = extract_json(text)
        except  Exception as inst:
            rospy.loginfo(inst)
            return None

        rospy.loginfo(json_dict)
        action = None
        if "action" in json_dict:
            action = json_dict["action"]

        return action

    def ee_move(self, action):

        #Servo in EE base_link frame
        move = False
        x = 0.0
        y = 0.0
        z = 0.0
        roll = 0.0
        yaw = 0.0
        pitch = 0.0

        rospy.loginfo(f"ee move: {action}")

        if "PITCH_UP" in action:
            rospy.loginfo("PITCH_UP")
            pitch =-self.angular_speed
            move = True
        elif "PITCH_DOWN" in action:
            rospy.loginfo("PITCH_DOWN")
            pitch = self.angular_speed
            move = True

        if  "ROLL_LEFT" in action:
            rospy.loginfo("ROLL_LEFT")
            roll =-self.angular_speed
            move = True
        elif "ROLL_RIGHT" in action:
            rospy.loginfo("ROLL_RIGHT")
            roll = self.angular_speed
            move = True
        
        if "YAW_LEFT" in action:
            rospy.loginfo("YAW_LEFT")
            yaw =-self.angular_speed
            move = True
        elif "YAW_RIGHT" in action:
            rospy.loginfo("YAW_RIGHT")
            yaw = self.angular_speed
            move = True
    
        if "MOVE_FORWARD" in action:
            rospy.loginfo("MOVE_FORWARD")
            x = self.speed
            move = True
        elif "MOVE_BACKWARD" in action:
            rospy.loginfo("MOVE_BACKWARD")
            x =-self.speed
            move = True

        if "MOVE_RIGHT" in action:
            rospy.loginfo("MOVE_RIGHT")
            y =-self.speed
            move = True
        elif "MOVE_LEFT" in action:
            rospy.loginfo("MOVE_LEFT")
            y = self.speed
            move = True
        
        if "MOVE_UP" in action:
            rospy.loginfo("MOVE_UP")
            z = self.speed
            move = True
        elif "MOVE_DOWN" in action:
            rospy.loginfo("MOVE_DOWN")
            z =-self.speed
            move = True
        
        if move:
            self.send_command(x,y,z, roll, pitch, yaw)

        if "CLOSE_HAND" in action:
            self.close()
        if "OPEN_HAND" in action:
            self.open()            

    def send_command(self, x, y, z, roll, pitch, yaw):
        linear_cmd = TwistStamped()
        linear_cmd.header.frame_id ="right_base_link"
        linear_cmd.twist.linear.x = x
        linear_cmd.twist.linear.y = y
        linear_cmd.twist.linear.z = z

        angular_cmd = TwistStamped()
        angular_cmd.header.frame_id ="right_end_effector_link"
        angular_cmd.twist.angular.x = pitch
        angular_cmd.twist.angular.y = yaw
        angular_cmd.twist.angular.z = roll

        rospy.loginfo(f"\ntranslate x: {x}, y: {y}, z:{z} \nrotate: roll: {roll}, pitch: {pitch}, yaw: {yaw}")

        #splitting up rotation from translation for now
        # try and add in transform from ee frame to base
        # to do both at same time
        #Hypothesise people will find this easier to convey
        # rotation in ee frame compared to base

        if (x != 0.0 or y != 0.0 or z != 0.0):
            rospy.loginfo(f"translate")
            for i in range(self.num_msgs):
                linear_cmd.header.stamp = rospy.Time.now()
                self.cart_vel_pub.publish(linear_cmd)
                self.rate.sleep()

        self.send_zero_twist_cmd()

        if (roll != 0.0 or pitch != 0.0 or yaw != 0.0):
            rospy.loginfo(f"rotate")
            for i in range(self.num_msgs):
                angular_cmd.header.stamp = rospy.Time.now()
                self.cart_vel_pub.publish(angular_cmd)
                self.rate.sleep()

        self.send_zero_twist_cmd()

    def send_zero_twist_cmd(self):
        zero_cmd = TwistStamped()
        zero_cmd.header.frame_id ="right_end_effector_link"
        zero_cmd.twist.linear.x = 0
        zero_cmd.twist.linear.y = 0
        zero_cmd.twist.linear.z = 0
        zero_cmd.twist.angular.x = 0
        zero_cmd.twist.angular.y = 0
        zero_cmd.twist.angular.z = 0

        for i in range(4):
            zero_cmd.header.stamp = rospy.Time.now()
            self.cart_vel_pub.publish(zero_cmd)
            self.rate.sleep()

    def indicate(self, position):
        rospy.loginfo(f"indicate:{position.header.frame_id} {position.point.x}, {position.point.y}, {position.point.z}")
        self.open()
        stamped_pose = PoseStamped()
        stamped_pose.header = position.header
        stamped_pose.pose.position = deepcopy(position.point)
        stamped_pose.pose.orientation.x = -1
        stamped_pose.pose.orientation.w = 0
        stamped_pose.pose.position.z += 0.125
        self.debug_pose_pub.publish(stamped_pose)
        self.right_arm_move_to_pose(stamped_pose)
        self.close()

    def grab(self, position):
        rospy.loginfo(f"grab:{position.header.frame_id} {position.point.x}, {position.point.y}, {position.point.z}")
        final_pose = PoseStamped()
        final_pose.header = position.header
        final_pose.pose.position = deepcopy(position.point)
        final_pose.pose.orientation.x = -1
        final_pose.pose.orientation.w = 0
        final_pose.pose.position.z -= 0.03

        min_safe_height = 0.065
        final_pose.pose.position.z = max(min_safe_height, final_pose.pose.position.z)

        self.open()
        self.debug_pose_pub.publish(final_pose)
        self.right_arm_move_to_pose(final_pose)
        self.close()

        retreat_pose = PoseStamped()
        retreat_pose.header = position.header
        retreat_pose.pose.position = deepcopy(position.point)
        retreat_pose.pose.orientation.x = -1
        retreat_pose.pose.orientation.w = 0
        retreat_pose.pose.position.z += 0.125
        self.debug_pose_pub.publish(retreat_pose)
        self.right_arm_move_to_pose(retreat_pose)

        init_pose = PoseStamped()
        init_pose.header.frame_id = "world"
        init_pose.pose.position.x =  0.3
        init_pose.pose.position.y = -0.4
        init_pose.pose.position.z =  0.4
        init_pose.pose.orientation.x = -1.0
        init_pose.pose.orientation.y =  0.0
        init_pose.pose.orientation.z =  0.0
        init_pose.pose.orientation.w =  0.0
        self.right_arm_move_to_pose(init_pose)

    def right_arm_move_to_pose(self, pose):
        rospy.wait_for_service('/my_gen3_right/move_pose')
        print("right_arm_move_to_pose")
        try:
            moveit_pose = rospy.ServiceProxy('/my_gen3_right/move_pose', MoveITPose)
            resp = moveit_pose(pose)
            return resp
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s"%e)

    def close(self):
        service_name = "/my_gen3_right/close_hand"
        rospy.wait_for_service(service_name)
        print(service_name)
        try:
            close_hand = rospy.ServiceProxy(service_name, Trigger)
            resp = close_hand()
            return resp
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s"%e)
    
    def open(self):
        service_name = "/my_gen3_right/open_hand"
        rospy.wait_for_service(service_name)
        print(service_name)
        try:
            open_hand = rospy.ServiceProxy(service_name, Trigger)
            resp = open_hand()
            return resp
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s"%e)

    def transform_positions(self, positions, org_frame, new_frame):
        t = rospy.Time.now()
        self.listener.waitForTransform(org_frame, new_frame, t, rospy.Duration(4.0))
        new_positions = []
        for p in positions:
            stamped_p = PointStamped()
            stamped_p.header.frame_id = org_frame
            stamped_p.header.stamp = t
            stamped_p.point = p
            new_p = self.listener.transformPoint(new_frame, stamped_p)

            new_positions.append(new_p)

        return new_positions

if __name__ == '__main__':
    llm = AssemblyClient()