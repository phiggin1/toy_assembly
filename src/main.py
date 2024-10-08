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
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger 
from std_msgs.msg import String
import pandas
import os
import time
import cv2

def extract_json(text):
    a = text.find('{')
    b = text.find('}')+1
    text_json = text[a:b]
    try:
        json_dict = json.loads(text_json)
        return json_dict
    except:
        return None

class AssemblyClient:
    def __init__(self):
        rospy.init_node('toy_assembly_main')
        self.cvbridge = CvBridge()
        self.listener = tf.TransformListener()

        self.prefix =  rospy.get_param("~prefix", "test")
        path = "/home/rivr/toy_logs"
        os.makedirs(path, exist_ok=True)
        start_time = time.strftime("%Y_%m_%d_%H_%M")
        self.log_file_path = os.path.join(path, f"{self.prefix}_{start_time}.csv")
        rospy.loginfo(self.log_file_path)
        
        self.dataframe_csv = []

        self.num_msgs = 20
        self.rate = rospy.Rate(20)
        self.speed = 0.1
        self.angular_speed = 1.0
        
        self.prev = None
        self.state = "LOW_LEVEL"
        self.check = False
        self.debug = rospy.get_param("~debug", True)
        server_port = rospy.get_param("~port", "8877")

        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % server_port)
        rospy.loginfo(f"Server listening on port:{server_port}")

        move_service_name = "/my_gen3_right/move_pose"
        rospy.wait_for_service(move_service_name)
        self.moveit_pose = rospy.ServiceProxy(move_service_name, MoveITPose)

        open_service_name = "/my_gen3_right/open_hand"
        rospy.wait_for_service(open_service_name)
        self.open_hand = rospy.ServiceProxy(open_service_name, Trigger)

        close_service_name = "/my_gen3_right/close_hand"
        rospy.wait_for_service(close_service_name)
        self.close_hand = rospy.ServiceProxy(close_service_name, Trigger)

        self.llm_text_srv = rospy.ServiceProxy('/gpt_servcice', LLMImage)
        self.llm_context_srv = rospy.ServiceProxy('/gpt_context', LLMImage)

        self.robot_speech_pub = rospy.Publisher('/text_to_speech', String, queue_size=10)
        self.debug_pose_pub = rospy.Publisher('/0debug_pose', PoseStamped, queue_size=10)
        self.status_pub = rospy.Publisher("/status", String, queue_size=10)

        self.twist_topic  = "/my_gen3_right/workspace/delta_twist_cmds"
        self.cart_vel_pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=10)
        
        #self.text_sub = rospy.Subscriber("/transcript", Transcription, self.text_cb)
        #rospy.spin()


        '''
        img = rospy.wait_for_message("//unity/camera/left/rgb/image_raw", Image, timeout=10)
        req = LLMImageRequest()
        req.image = img
        resp = self.llm_context_srv(req)
        print(resp)
        return

        rospy.loginfo("waiting for objects")
        try:        
            obj_img = rospy.wait_for_message("/left_object_images", ObjectImage, timeout=10)
        except:
            rospy.loginfo("object waiting timed out")
            return
        
        opject_positions = obj_img.object_positions
        objects = []
        for i in range(len(opject_positions)):
            objects.append(f"obj_{i}")
        req = LLMImageRequest()
        req.objects = objects
        req.image = obj_img.image
        resp = self.llm_context_srv(req)
        
        print(resp)
        return
        '''
        
        rospy.on_shutdown(self.shutdown_hook)
        while not rospy.is_shutdown():
            '''
            transcription = rospy.wait_for_message("/transcript", Transcription)
            '''
            text = input("command: ")
            transcription = Transcription()
            transcription.transcription = text
            
            self.text_cb(transcription)

    def shutdown_hook(self):
        if len(self.dataframe_csv) > 0:
            pandas.concat(self.dataframe_csv).to_csv(self.log_file_path, index=False)

    def text_cb(self, transcript):
        if self.debug: rospy.loginfo("========================================================") 
        self.df = pandas.DataFrame({
            "timestamp": [transcript.audio_recieved],
            "duration": [transcript.duration],
            "transcript":[transcript.transcription]
        })
        rospy.loginfo("THINKING")
        self.status_pub.publish("THINKING")
        if self.debug: rospy.loginfo(f"state: {self.state}")
        if self.debug: rospy.loginfo(f"audio transcript: {transcript}")
        if self.debug: rospy.loginfo(f"prev command|action: {self.prev}")

        transcript =  transcript.transcription

        if self.state == "HIGH_LEVEL":
            results = self.high_level(transcript)
        else:
            results = self.low_level(transcript)
        
        self.prev = (transcript, results[0] if results is not None else None)
        self.df["results"] = [results]

        rospy.loginfo(results)
        rospy.loginfo("WAITING")
        self.status_pub.publish("WAITING")
        if self.debug: rospy.loginfo("--------------------------------------------------------") 
        self.dataframe_csv.append(self.df)


    def high_level(self, text):
        rospy.loginfo("waiting for objects")
        try:        
            obj_img = rospy.wait_for_message("/left_object_images", ObjectImage, timeout=10)
        except:
            rospy.loginfo("object waiting timed out")
            self.state = "LOW_LEVEL"
            return None
        rospy.loginfo("recved objects")
        
        frame = obj_img.header.frame_id
        image = obj_img.image
        opject_positions = obj_img.object_positions

        objects = []
        for i in range(len(opject_positions)):
            objects.append(f"obj_{i}")

        obj_pos = []
        for i in range(len(opject_positions)):
            rospy.loginfo(f"{objects[i]}, {opject_positions[i].header.frame_id}, {opject_positions[i].point.x}, {opject_positions[i].point.y},{opject_positions[i].point.z}")
            obj_pos.append([opject_positions[i].point.x, opject_positions[i].point.y, opject_positions[i].point.z])

        req = LLMImageRequest()
        req.text = text
        req.objects = objects
        req.check = self.check
        req.image = image

        resp = self.llm_text_srv(req)

        cv_img = self.cvbridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        merge_test = "_".join(text.split(" "))
        fname = f"/home/rivr/toy_logs/images/{image.header.stamp}{merge_test}.png"
        print(fname)
        cv2.imwrite(fname, cv_img)

        self.df["image_path"] = [fname]
        self.df["objects"] = [objects]
        self.df["objects_positions"] = [obj_pos]
        self.df["gpt_response"] = [str(resp.text).replace('\n','')]

        #Should do some error checking
        # in future
        json_dict = extract_json(resp.text)
        rospy.loginfo(f"init json dict:\n{json_dict}")

        if json_dict is None:
            self.state = "LOW_LEVEL"
            return None

        action = None
        if "action" in json_dict:
            action = json_dict["action"]
        else:
            self.state = "LOW_LEVEL"
            return None
    
        results = None
        if "PICKUP" in action or "PICK_UP" in action:
            if len(objects) > 0:
                success = self.pickup(json_dict, objects, opject_positions)
                results = ("PICKUP", success)
            else:
                results = ("PICKUP", False)
        elif "MOVE_TO" in action:
            if len(objects) > 0:
                print(json_dict)
                if "object" in json_dict:
                    print(json_dict["object"])
                    obj = json_dict["object"]
                    indx = objects.index(obj)
                    self.target_position = opject_positions[indx]
                    print(f"{obj}, {self.target_position}")
                    success = self.move_to(self.target_position)
                    results = ("MOVE_TO", success)
                else:
                    results = ("MOVE_TO", False)
            else:
                results = ("MOVE_TO", False)

        else:   
            any_valid_commands = self.ee_move(action)
            results = (action, any_valid_commands)

        self.state = "LOW_LEVEL"
        return results

    def low_level(self, text):
        action = self.send_ada(text)
        rospy.loginfo(f"low level action:\n\t {action}")
        results = None
        if action is None or len(action)<1:
            self.state = "HIGH_LEVEL"
            results = self.high_level(text)  
            return
        
        if "NO_ACTION" in action:
            rospy.loginfo("No action")
            return
        
        any_valid_commands = False

        if "PICKUP" in action or  "PICK_UP" in action or"OTHER" in action or  "MOVE_TO" in action:
            any_valid_commands = True
            self.state = "HIGH_LEVEL"
            results = self.high_level(text)
        elif ("MOVE_UP" in action and "MOVE_DOWN" in action ) or ("MOVE_LEFT" in action and "MOVE_RIGHT" in action) or ("MOVE_FORWARD" in action and "MOVE_BACKWARD" in action) or ("PITCH_UP" in action and "PITCH_DOWN" in action ) or ("ROLL_LEFT" in action and "ROLL_RIGHT" in action):
            any_valid_commands = True
            self.state = "HIGH_LEVEL"
            results = self.high_level(text)
        else:   
            rospy.loginfo(f"state: {self.state}")
            self.state = "LOW_LEVEL"
            any_valid_commands = self.ee_move(action)
            results = (action, any_valid_commands)

        if not any_valid_commands:
            self.state = "HIGH_LEVEL"
            results = self.high_level(text)

        return results


    def send_ada(self, text):
        msg = {"type":"llm",
               "text":text,
               "prev":self.prev
        }

        if self.debug: rospy.loginfo(f"LLM sending to ada\ntext:{text}")
        
        self.socket.send_json(msg)
        resp = self.socket.recv_json()

        self.df["phi3_response"] = [str(resp["text"]).replace('\n','')]

        if "error" in resp:
            rospy.loginfo(resp["error"])
            return None

        text_resp = resp["text"]
        rospy.loginfo(f"response:\n{text_resp}\n ---")
        
        try:
            json_dict = extract_json(text_resp)
        except  Exception as inst:
            rospy.loginfo(inst)
            return None

        rospy.loginfo(json_dict)

        if json_dict is None:
            return None
        
        action = None
        if "action" in json_dict:
            action = json_dict["action"]
    
        return action
    
    def pickup(self, json_dict, objects, opject_positions):
        open_succes = self.open()
        rospy.loginfo(f"open_succes: {open_succes}")
        if not open_succes:
            return open_succes
        
        obj = json_dict["object"]
        indx = objects.index(obj)
        self.target_position = opject_positions[indx]
        init_grab_move_to_success = self.move_to(self.target_position)
        rospy.loginfo(f"init grab move_to successful : {init_grab_move_to_success}")
        if not init_grab_move_to_success:
            return init_grab_move_to_success

        grab_success = self.grab(self.target_position)
        #Reset the state
        self.state = "LOW_LEVEL"
        return grab_success

    def ee_move(self, actions):
        any_valid_commands = False
        #Servo in EE base_link frame
        move = False
        x = 0.0
        y = 0.0
        z = 0.0
        roll = 0.0
        yaw = 0.0
        pitch = 0.0

        rospy.loginfo(f"ee move: {actions}")

        for action in actions:
            if ("PITCH_UP" in action or "PITCH_DOWN" in action or "ROLL_LEFT" in action or "ROLL_RIGHT" in action or "YAW_LEFT" in action or "YAW_RIGHT" in action or 
                "MOVE_FORWARD" in action or "MOVE_BACKWARD" in action or "MOVE_RIGHT" in action or "MOVE_LEFT" in action or "MOVE_UP" in action or "MOVE_DOWN" in action):
                if "PITCH_UP" in action:
                    rospy.loginfo("PITCH_UP")
                    pitch =-self.angular_speed
                    move = True
                    any_valid_commands = True
                elif "PITCH_DOWN" in action:
                    rospy.loginfo("PITCH_DOWN")
                    pitch = self.angular_speed
                    move = True
                    any_valid_commands = True

                if  "ROLL_LEFT" in action:
                    rospy.loginfo("ROLL_LEFT")
                    roll =-self.angular_speed
                    move = True
                    any_valid_commands = True
                elif "ROLL_RIGHT" in action:
                    rospy.loginfo("ROLL_RIGHT")
                    roll = self.angular_speed
                    move = True
                    any_valid_commands = True
                
                if "YAW_LEFT" in action:
                    rospy.loginfo("YAW_LEFT")
                    yaw =-self.angular_speed
                    move = True
                    any_valid_commands = True
                elif "YAW_RIGHT" in action:
                    rospy.loginfo("YAW_RIGHT")
                    yaw = self.angular_speed
                    move = True
                    any_valid_commands = True
            
                if "MOVE_FORWARD" in action:
                    rospy.loginfo("MOVE_FORWARD")
                    x = self.speed
                    move = True
                    any_valid_commands = True
                elif "MOVE_BACKWARD" in action:
                    rospy.loginfo("MOVE_BACKWARD")
                    x =-self.speed
                    move = True
                    any_valid_commands = True

                if "MOVE_RIGHT" in action:
                    rospy.loginfo("MOVE_RIGHT")
                    y =-self.speed
                    move = True
                    any_valid_commands = True
                elif "MOVE_LEFT" in action:
                    rospy.loginfo("MOVE_LEFT")
                    y = self.speed
                    move = True
                    any_valid_commands = True
                
                if "MOVE_UP" in action:
                    rospy.loginfo("MOVE_UP")
                    z = self.speed
                    move = True
                    any_valid_commands = True
                elif "MOVE_DOWN" in action:
                    rospy.loginfo("MOVE_DOWN")
                    z =-self.speed
                    move = True
                    any_valid_commands = True
            else:
                if "CLOSE_HAND" in action:
                    any_valid_commands = True
                    if move:
                        self.send_command(x,y,z, roll, pitch, yaw)
                        move = False
                        x = 0.0
                        y = 0.0
                        z = 0.0
                        roll = 0.0
                        yaw = 0.0
                        pitch = 0.0
                    self.close()
                if "OPEN_HAND" in action:
                    any_valid_commands = True
                    if move:
                        self.send_command(x,y,z, roll, pitch, yaw)
                        move = False
                        x = 0.0
                        y = 0.0
                        z = 0.0
                        roll = 0.0
                        yaw = 0.0
                        pitch = 0.0
                    self.open()  
        if move:
            self.send_command(x,y,z, roll, pitch, yaw)
            move = False

        return any_valid_commands 

    def send_command(self, x, y, z, roll, pitch, yaw):
        linear_cmd = TwistStamped()
        
        linear_cmd.header.frame_id ="right_base_link"
        linear_cmd.twist.linear.x = x
        linear_cmd.twist.linear.y = y
        linear_cmd.twist.linear.z = z
        
        '''
        linear_cmd.header.frame_id ="right_end_effector_link"
        linear_cmd.twist.linear.x = z
        linear_cmd.twist.linear.y = -y
        linear_cmd.twist.linear.z = x
        '''

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

    def move_to(self, position):
        rospy.loginfo(f"move_to:{position.header.frame_id} {position.point.x}, {position.point.y}, {position.point.z}")
        stamped_pose = PoseStamped()
        stamped_pose.header = position.header
        stamped_pose.pose.position = deepcopy(position.point)
        stamped_pose.pose.orientation.x = -1
        stamped_pose.pose.orientation.w = 0
        stamped_pose.pose.position.z += 0.125
        self.debug_pose_pub.publish(stamped_pose)
        status = self.right_arm_move_to_pose(stamped_pose)
        rospy.loginfo(f"move_to successful : {status}")
        return status

    def grab(self, position):
        rospy.loginfo(f"grab:{position.header.frame_id} {position.point.x}, {position.point.y}, {position.point.z}")
        final_pose = PoseStamped()
        final_pose.header = position.header
        final_pose.pose.position = deepcopy(position.point)
        final_pose.pose.orientation.x = -1
        final_pose.pose.orientation.w = 0
        final_pose.pose.position.z -= 0.075

        min_safe_height = 0.065
        final_pose.pose.position.z = max(min_safe_height, final_pose.pose.position.z)

        open_success = self.open()
        rospy.loginfo(f"open successful : {open_success}")
        if not open_success:
            return open_success
        
        self.debug_pose_pub.publish(final_pose)
        move_to_grab_success = self.right_arm_move_to_pose(final_pose)
        rospy.loginfo(f"move to grab successful : {move_to_grab_success}")
        if not move_to_grab_success:
            return move_to_grab_success
        
        close_success = self.close()
        rospy.loginfo(f"close successful : {close_success}")
        if not close_success:
            return close_success
        

        retreat_pose = PoseStamped()
        retreat_pose.header = position.header
        retreat_pose.pose.position = deepcopy(position.point)
        retreat_pose.pose.orientation.x = -1
        retreat_pose.pose.orientation.w = 0
        retreat_pose.pose.position.z += 0.125
        self.debug_pose_pub.publish(retreat_pose)
        retreat_pose_success = self.right_arm_move_to_pose(retreat_pose)
        rospy.loginfo(f"retreat successful : {retreat_pose_success}")

        return retreat_pose_success

    def right_arm_move_to_pose(self, pose):
        try:
            success = False
            count = 1
            num_retries = 3
            while count < num_retries and not success:
                resp = self.moveit_pose(pose)
                count += 1
                success = resp.result
            rospy.loginfo(f"right_arm_move_to_pose: {count} tries before {success}")
            return success
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s"%e)
            return False

    def close(self):
        try:
            resp = self.close_hand()
            return resp.success
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s"%e)
            return False
    
    def open(self):
        try:
            resp = self.open_hand()
            return resp.success
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s"%e)
            return False

    def transform_positions(self, positions, org_frame, new_frame):
        t = rospy.Time.now()
        self.listener.waitForTransform(org_frame, new_frame, t, rospy.Duration(4.0))
        new_positions = []
        for p in positions:
            stamped_p = PointStamped()
            stamped_p.header.frame_id = org_frame
            stamped_p.header.stamp = t
            stamped_p.point.x = p.point.x
            stamped_p.point.y = p.point.y
            stamped_p.point.z = p.point.z
            new_p = self.listener.transformPoint(new_frame, stamped_p)

            new_positions.append(new_p)

        return new_positions

if __name__ == '__main__':
    llm = AssemblyClient()