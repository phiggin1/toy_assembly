#!/usr/bin/env python3


import os
import tf
import cv2
import time
import json
import struct
import pandas
import numpy as np
import rospy
import message_filters
from cv_bridge import CvBridge
from multiprocessing import Lock
from copy import deepcopy
from image_geometry import PinholeCameraModel
from toy_assembly.msg import Detection
import pycocotools.mask as mask_util
from toy_assembly.msg import Transcription, ObjectImage
from toy_assembly.srv import LLMImage, LLMImageRequest
from toy_assembly.srv import LLMText, LLMTextRequest
from toy_assembly.srv import SAM, SAMRequest, SAMResponse
from toy_assembly.srv import ObjectLocation, ObjectLocationRequest, ObjectLocationResponse
from toy_assembly.srv import MoveITGrabPose, MoveITGrabPoseRequest, MoveITGrabPoseResponse
from toy_assembly.srv import MoveITPose
from std_msgs.msg import String
from std_msgs.msg import Header
from std_srvs.srv import Trigger 
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TwistStamped, PoseStamped, Point, PointStamped

def extract_json(text):
    a = text.find('{')
    b = text.rfind('}')+1

    #print(a,b)

    text_json = text[a:b]

    #print(text_json)

    try:
        json_dict = json.loads(text_json)
        return json_dict
    except Exception as inst:
        print(type(inst))    # the exception type
        print(inst.args)     # arguments stored in .args
        print(inst)
        return None

def reject_outliers(data, m=20):
    d = np.abs(data[:,0] - np.median(data[:,0]))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)

    resp = data[s < m]

    return resp

def create_cloud(points, frame):
    point_list = []
    for p in points:
        x = p[0]
        y = p[1]
        z = p[2]
        pt = [x, y, z, 0]
        r = p[3]
        g = p[4]
        b = p[5]
        a = 255
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        pt[3] = rgb
        point_list.append(pt)

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 16, PointField.UINT32, 1),
    ]

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame
    pc2 = point_cloud2.create_cloud(header, fields, point_list)

    return pc2

def transform_point(ps, mat44, target_frame):
    xyz = tuple(np.dot(mat44, np.array([ps.point.x, ps.point.y, ps.point.z, 1.0])))[:3]
    r = PointStamped()
    r.header.stamp = ps.header.stamp
    r.header.frame_id = target_frame
    r.point = Point(*xyz)

    return r

class AssemblyClient:
    def __init__(self):
        rospy.init_node('toy_assembly_main')
        self.cvbridge = CvBridge()
        self.listener = tf.TransformListener()
        self.mutex = Lock()

        self.prefix =  str(rospy.get_param("~prefix", "test"))
        path = "/home/rivr/toy_logs"
        os.makedirs(os.path.join(path, self.prefix), exist_ok=True)
        os.makedirs(os.path.join(path, self.prefix, 'images'), exist_ok=True)
        start_time = time.strftime("%Y_%m_%d_%H_%M")
        name = f"{self.prefix}_{start_time}.csv"
        self.log_file_path = os.path.join(path, self.prefix, name)
        rospy.loginfo(self.log_file_path)
        
        self.dataframe_csv = []

        self.num_msgs = 20 
        self.rate = rospy.Rate(20)
        self.speed = 0.1
        self.angular_speed = 1.0
        
        self.prev = None
        self.debug = rospy.get_param("~debug", True)
        self.sim_time = rospy.get_param("/use_sim_time")#, False)

        rospy.loginfo(f"sim time active: {self.sim_time}")
        
        grab_cloud_service_name = "/my_gen3_right/grab_object"
        rospy.wait_for_service(grab_cloud_service_name)
        self.grab_cloud_srv = rospy.ServiceProxy(grab_cloud_service_name, MoveITGrabPose)
        print(grab_cloud_service_name)

        move_service_name = "/my_gen3_right/move_pose"
        rospy.wait_for_service(move_service_name)
        self.moveit_pose = rospy.ServiceProxy(move_service_name, MoveITPose)

        open_service_name = "/my_gen3_right/open_hand"
        rospy.wait_for_service(open_service_name)
        self.open_hand = rospy.ServiceProxy(open_service_name, Trigger)

        close_service_name = "/my_gen3_right/close_hand"
        rospy.wait_for_service(close_service_name)
        self.close_hand = rospy.ServiceProxy(close_service_name, Trigger)

        gpt_service_name = "/gpt_servcice"
        rospy.wait_for_service(gpt_service_name)
        self.llm_image_srv = rospy.ServiceProxy(gpt_service_name, LLMImage)
        print(gpt_service_name)

        phi_service_name = "/phi_servcice"
        rospy.wait_for_service(phi_service_name)
        self.llm_text_srv = rospy.ServiceProxy(phi_service_name, LLMText)
        print(phi_service_name)

        sam_service_name = "/get_sam_segmentation"
        rospy.wait_for_service(sam_service_name)
        self.sam_srv = rospy.ServiceProxy(sam_service_name, SAM)
        print(sam_service_name)

        #gpt_state_service_name = "/gpt_state_servcice"
        #rospy.wait_for_service(gpt_state_service_name)
        #self.llm_state_srv = rospy.ServiceProxy(gpt_state_service_name, LLMImage)

        gpt_part_location_service_name = "/gpt_part_location"
        rospy.wait_for_service(gpt_part_location_service_name)
        self.part_location = rospy.ServiceProxy(gpt_part_location_service_name, ObjectLocation)
        print(gpt_part_location_service_name)


        self.twist_topic  = "/my_gen3_right/workspace/delta_twist_cmds"
        self.cart_vel_pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=10)
        self.robot_speech_pub = rospy.Publisher('/text_to_speech', String, queue_size=10)
        self.status_pub = rospy.Publisher("/status", String, queue_size=10)

        self.debug_pose_pub = rospy.Publisher('/0debug/pose', PoseStamped, queue_size=10)
        self.test_cloud = rospy.Publisher("/0debug/test_cloud", PointCloud2, queue_size=10)

        fp_init_env = "/home/rivr/toy_ws/src/toy_assembly/prompts/init_env.txt"
        with open(fp_init_env) as f:
            self.init_env = f.read()

        self.env = self.init_env

        self.colors = {
            "tan tray":(255,0,0),
            "orange tray":(0,255,0),
            "tan horse body":(0,0,255),
            "blue horse legs":(255,255,0),
            "orange horse legs":(255,0,255),
            "table":(255,255,255)
        }
        
        if not self.sim_time:
            self.cam_link_name = f"left_camera_color_frame"
            cam_info_topic = f"/left_camera/color/camera_info_throttled"
            rgb_image_topic = f"/left_camera/color/image_rect_color_throttled"              
            depth_image_topic = f"/left_camera/depth_registered/sw_registered/image_rect_throttled"
        else:
            self.cam_link_name = f"left_camera_link"
            cam_info_topic = f"/unity/camera/left/rgb/camera_info"
            rgb_image_topic = f"/unity/camera/left/rgb/image_raw"
            depth_image_topic = f"/unity/camera/left/depth/image_raw"

        rospy.loginfo(cam_info_topic)
        rospy.loginfo(rgb_image_topic)
        rospy.loginfo(depth_image_topic)

        self.cam_info = rospy.wait_for_message(cam_info_topic, CameraInfo)
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(self.cam_info)
        self.rgb_sub = message_filters.Subscriber(rgb_image_topic, Image)
        self.depth_sub = message_filters.Subscriber(depth_image_topic, Image)
        ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], slop=0.2, queue_size=10)
        ts.registerCallback(self.image_cb)
        self.rgb_image = None
        self.depth_image = None
        self.have_images = False
        
        self.init_objects = [
            "tan tray",
            "orange tray",
            "tan horse body",
            "blue horse legs",
            "orange horse legs",
            "table"
        ]

        rate = rospy.Rate(5)
        while not self.have_images:
            rate.sleep()

        rospy.on_shutdown(self.shutdown_hook)
        while not rospy.is_shutdown():
            
            transcription = rospy.wait_for_message("/transcript", Transcription)
            self.text_cb(transcription)
            '''
            text = input("command: ")
            transcription = Transcription()
            transcription.transcription = text
            self.text_cb(transcription)
            '''
            

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
        if self.debug: rospy.loginfo(f"audio transcript: '{transcript.transcription}'")
        if self.debug: rospy.loginfo(f"prev command|action: {self.prev}")

        transcript =  transcript.transcription

        #results = self.high_level(transcript)
        results = self.low_level(transcript)
        
        self.prev = (transcript, results[0] if results is not None else None)
        self.df["results"] = [results]

        rospy.loginfo(f"results: {results}")
        #print(f"env: {self.env}")

        '''
        #check if env makes sense with what is seen
        if results is not None:
            self.env = self.check_env(results[0], results[1], self.env)
        '''

        rospy.loginfo("WAITING")
        self.status_pub.publish("WAITING")
        if self.debug: rospy.loginfo("--------------------------------------------------------") 
        self.dataframe_csv.append(self.df)

      

    def image_cb(self, rgb_ros_image, depth_ros_image):
        success = self.mutex.acquire(block=False, timeout=0.02)
        if success:
            #print(f"rgb eoncoding: {rgb_ros_image.encoding}")
            #print(f"depth eoncoding: {depth_ros_image.encoding}")
            self.rgb_encoding = rgb_ros_image.encoding
            self.depth_encoding = depth_ros_image.encoding
            self.rgb_image = self.cvbridge.imgmsg_to_cv2(rgb_ros_image, desired_encoding="bgr8") 
            self.depth_image = self.cvbridge.imgmsg_to_cv2(depth_ros_image)
            if not self.have_images:
                rospy.loginfo("HAVE IMAGES")
                self.have_images = True
            self.mutex.release()

    def check_env(self, action, success, env):
        text = {
            "action":action,
            "success":success
        }
        image, objects, _, _, _ = self.get_detections(". ".join(self.init_objects)+'.')
            
        with self.mutex:
            req = LLMImageRequest()
            req.text = json.dumps(text)
            req.objects = objects
            req.env = self.init_env
            req.image = self.cvbridge.cv2_to_imgmsg(self.rgb_image, encoding="bgr8")
            resp = self.llm_state_srv(req)

            cv_img = self.cvbridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
            fname = os.path.join('/home', 'rivr', 'toy_logs', self.prefix, 'images', f"{image.header.stamp}_annotated.png")
            print(fname)
            cv2.imwrite(fname, cv_img)
            fname = os.path.join('/home', 'rivr', 'toy_logs', self.prefix, 'images', f"{image.header.stamp}.png")
            cv2.imwrite(fname, self.rgb_image)

            json_dict = extract_json(resp.text)

            if json_dict['correct']:
                print("predicted matches actual")
                return env
            else:
                act_env = json_dict["environment_actual"]
                print(f"predicted env:{env} does not match actual env: {act_env}")
                return json.dumps(json_dict["environment_actual"], indent=4)
    
    def log_images(self, text, stamp, annoated, rgb):
        cv_img = self.cvbridge.imgmsg_to_cv2(annoated, desired_encoding="passthrough")
        merge_test = "_".join(text.split(" "))
        print(f"Saving: {stamp}{merge_test}")
        fname = os.path.join('/home','rivr', 'toy_logs', self.prefix, 'images', f"{stamp}{merge_test}_annotated.png")#f"/home/rivr/toy_logs/images/{stamp}{merge_test}_annotated.png"
        
        cv2.imwrite(fname, cv_img)
        merge_test = "_".join(text.split(" "))
        fname = os.path.join('/home','rivr', 'toy_logs', self.prefix, 'images', f"{stamp}{merge_test}.png")#f"/home/rivr/toy_logs/images/{stamp}{merge_test}.png"

        cv2.imwrite(fname, rgb)

        return fname

    def high_level(self, text):
        rospy.loginfo("waiting for objects")
        image, objects, rles, bboxs, scores = self.get_detections("tan tray. orange tray. tan horse body. blue horse legs. orange horse legs. table. robot gripper. human hand.")
        
        with self.mutex:
            req = LLMImageRequest()
            req.text = text
            if self.env is None:
                self.env = self.init_env
            req.env = self.env
            req.objects = objects
            req.image = image

            resp = self.llm_image_srv(req)

        fname = self.log_images(text, image.header.stamp, image, self.rgb_image)
        self.df["image_path"] = [fname]
        self.df["objects"] = [objects]
        self.df["gpt_response"] = [str(resp.text).replace('\n','')]

        #Should do some error checking
        # in future
        json_dict = extract_json(resp.text)
        #rospy.loginfo(f"init json dict:\n{json_dict}")

        if json_dict is None:
            return ("NO_ACTION", True)

        action = None
        if "action" in json_dict:
            actions = json_dict["action"]
            print(f"actions:\n-----\n{actions}")
            print("-----")
        else:
            print("no actions")
            return ("NO_ACTION", True)
        
        results = []
        for i, action in enumerate(actions):
            print(f"action {i}: {action}")
            result = None
            if isinstance(action, str):
                result = ("NO_ACTION", True)
            elif "MOVE_TO" in action["action"]:
                if len(objects) > 0:
                    if "object" in action:
                        target_object = action["object"]
                        #get updated position if this is not the first action
                        '''if i > 0:
                            image, objects, rles, bboxs, scores = self.get_detections(target_object)
                            self.log_images(text, image.header.stamp, image, self.rgb_image)'''
                        resp = self.get_position(target_object, objects, rles, bboxs, scores)
                        if resp is not None:
                            target_position = resp[0]
                            print(f"{target_object}, {target_position.header.frame_id}, x:{target_position.point.x:.2f}, y:{target_position.point.y:.2f}, z:{target_position.point.z:.2f}")
                            success = self.move_to(target_position)
                            self.env = json.dumps(json_dict["environment_after"], indent=4)
                            result = ("MOVE_TO", success)
                        else:
                            result = ("MOVE_TO", False)
                    else:
                        result = ("MOVE_TO", False)
                else:
                    result = ("MOVE_TO", False)
            elif "PICKUP" in action["action"] or "PICK_UP" in action["action"]:
                if len(objects) > 0:
                    if "object" in action:
                        target_object = action["object"]
                        '''if i > 0:
                            image, objects, rles, bboxs, scores = self.get_detections(target_object)
                            self.log_images(text, image.header.stamp, image, self.rgb_image)'''
                        resp = self.get_position(target_object, objects, rles, bboxs, scores)
                        if resp is not None:
                            target_position = resp[0]
                            cloud = resp[1]
                            offset = None
                            if "location" in action:
                                resp = self.get_location_offset(target_object, objects, rles, bboxs, scores, text)
                                offset = resp.directions
                                print(f"offset: {offset}")
                            print(f"{target_object}, {target_position.header.frame_id}, x:{target_position.point.x:.2f}, y:{target_position.point.y:.2f}, z:{target_position.point.z:.2f}")
                            success = self.pickup(target_position, cloud, offset)
                            self.env = json.dumps(json_dict["environment_after"], indent=4)
                            result = ("PICKUP", success)
                        else:
                            result = ("PICKUP", False)
                else:
                    result = ("PICKUP", False)
            else:
                a = action["action"]
                any_valid_commands = self.ee_move([a])
                result = (a, any_valid_commands)
            results.append(result)
        return results

    def low_level(self, text):
        req = LLMTextRequest()
        req.text = text

        resp = self.llm_text_srv(req)

        self.df["phi3_response"] = [str(resp.text).replace('\n','')]
        action = self.parse_llm_response(resp.text)

        rospy.loginfo(f"low level action:\n\t {action}")
        results = None
        if action is None or len(action)<1:
            results = self.high_level(text)  
            return results
        
        if "NO_ACTION" in action:
            rospy.loginfo("No action")
            return ("NO_ACTION", True)
        
        any_valid_commands = False

        if "PICKUP" in action or  "PICK_UP" in action or"OTHER" in action or  "MOVE_TO" in action:
            any_valid_commands = True
            results = self.high_level(text)
        #check for contradictory commands
        elif ("MOVE_UP" in action and "MOVE_DOWN" in action ) or ("MOVE_LEFT" in action and "MOVE_RIGHT" in action) or ("MOVE_FORWARD" in action and "MOVE_BACKWARD" in action) or ("PITCH_UP" in action and "PITCH_DOWN" in action ) or ("ROTATE_LEFT" in action and "ROTATE_RIGHT" in action):
            any_valid_commands = True
            results = self.high_level(text)
        else:   
            rospy.loginfo(f"calling ee_move with actions: {action}")
            any_valid_commands = self.ee_move(action)
            results = (action, any_valid_commands)

        print(f"any_valid_commands:{any_valid_commands}")

        if not any_valid_commands:
            results = self.high_level(text)

        return results

    def parse_llm_response(self, text):
        print(f"\n+++++++++++\n{text}\n+++++++++++\n")
        json_dict = extract_json(text)
        if json_dict is None:
            return None
        
        action = None
        if "action" in json_dict:
            action = json_dict["action"]

        return action

    def pickup(self, target_position, cloud, offset=None):
        open_succes = self.open()
        rospy.loginfo(f"open_succes: {open_succes}")
        if not open_succes:
            return open_succes

        rospy.loginfo(f"pickup: {target_position.point.x},  {target_position.point.y}. {target_position.point.z}")
        init_grab_move_to_success = self.move_to(target_position)
        rospy.loginfo(f"init grab move_to successful : {init_grab_move_to_success}")
        if not init_grab_move_to_success:
            return init_grab_move_to_success

        grab_success = self.grab(target_position, cloud, offset)

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
        if type(actions) is not list: 
            actions = [actions]

        for action in actions:
            #check for valid actions
            if ("PITCH_UP" in action or "PITCH_DOWN" in action or "ROTATE_LEFT" in action or "ROTATE_RIGHT" in action or "YAW_LEFT" in action or "YAW_RIGHT" in action or 
                "MOVE_FORWARD" in action or "MOVE_BACKWARD" in action or "MOVE_RIGHT" in action or "MOVE_LEFT" in action or "MOVE_UP" in action or "MOVE_DOWN" in action):
                #check for specific actions
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

                if  "ROTATE_LEFT" in action:
                    rospy.loginfo("ROTATE_LEFT")
                    roll =-self.angular_speed
                    move = True
                    any_valid_commands = True
                elif "ROTATE_RIGHT" in action:
                    rospy.loginfo("ROTATE_RIGHT")
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
                        rospy.loginfo(f" CLOSE_HAND sending command: {x}, {y}, {z}, {roll}, {pitch}, {yaw}")

                        self.send_command(x,y,z, roll, pitch, yaw)
                        move = False
                        x = 0.0
                        y = 0.0
                        z = 0.0
                        roll = 0.0
                        yaw = 0.0
                        pitch = 0.0
                    self.close()
                elif "OPEN_HAND" in action:
                    any_valid_commands = True
                    if move:
                        rospy.loginfo(f" OPEN_HAND sending command: {x}, {y}, {z}, {roll}, {pitch}, {yaw}")
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
            rospy.loginfo(f"sending command: {x}, {y}, {z}, {roll}, {pitch}, {yaw}")
            self.send_command(x,y,z, roll, pitch, yaw)
            move = False
            any_valid_commands = True

        return any_valid_commands 

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

        rospy.loginfo(f"send_command translate x: {x}, y: {y}, z:{z} rotate: roll: {roll}, pitch: {pitch}, yaw: {yaw}")

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
                #rospy.loginfo(f"translate {i}")
            self.send_zero_twist_cmd()

        if (roll != 0.0 or pitch != 0.0 or yaw != 0.0):
            rospy.loginfo(f"rotate")
            for i in range(self.num_msgs*2):
                angular_cmd.header.stamp = rospy.Time.now()
                self.cart_vel_pub.publish(angular_cmd)
                self.rate.sleep()
                #rospy.loginfo(f"rotate {i}")
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
        stamped_pose = PoseStamped()
        stamped_pose.header = position.header
        stamped_pose.pose.position = deepcopy(position.point)
        stamped_pose.pose.orientation.x = -1
        stamped_pose.pose.orientation.w = 0
        stamped_pose.pose.position.z += 0.125
        rospy.loginfo(f"move_to: {stamped_pose.header.frame_id} {stamped_pose.pose.position.x:.3f}, {stamped_pose.pose.position.y:.3f}, {stamped_pose.pose.position.z:.3f}")

        status = self.right_arm_move_to_pose(stamped_pose)
        rospy.loginfo(f"move_to successful : {status}")
        return status

    def grab(self, position, cloud, offset=None):
        rospy.loginfo(f"grab: {position.header.frame_id} {position.point.x:.3f}, {position.point.y:.3f}, {position.point.z:.3f}")
        print(cloud.header.frame_id)
        grab = MoveITGrabPoseRequest()
        grab.cloud = cloud
        if offset is not None:
            grab.offsets = offset
        
        resp = self.grab_cloud_srv(grab)

        return resp.result
    
        '''
        final_pose = PoseStamped()
        final_pose.header = position.header
        final_pose.pose.position = deepcopy(position.point)
        final_pose.pose.orientation.x = -1
        final_pose.pose.orientation.w = 0
        final_pose.pose.position.z -= 0.03725

        self.min_safe_height = 0.065
        final_pose.pose.position.z = max(self.min_safe_height, final_pose.pose.position.z)

        open_success = self.open()
        rospy.loginfo(f"open successful : {open_success}")
        if not open_success:
            return open_success
        
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
        
        retreat_pose_success = self.right_arm_move_to_pose(retreat_pose)
        rospy.loginfo(f"retreat successful : {retreat_pose_success}")
        
        return retreat_pose_success
        '''

    def right_arm_move_to_pose(self, pose):
        self.debug_pose_pub.publish(pose)
        rospy.loginfo(f"right_arm_move_to_pose: {pose.header.frame_id} {pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, {pose.pose.position.z:.3f}")
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

    def get_detections(self, text):
        req = SAMRequest()
        req.text_prompt = text
        resp = self.sam_srv(req)

        annotated_img = resp.annotated_image
        raw_img = resp.image
        objects = []
        rles = []
        bboxes = []
        scores = []
        for obj in resp.object:
            objects.append(obj.class_name)
            rles.append(obj.rle_encoded_mask)
            box = [obj.u_min, obj.v_min,obj.u_max, obj.v_max]
            bboxes.append(box)
            scores.append(obj.score)

        return (annotated_img, objects, rles, bboxes, scores)
    
    def get_position(self, target_object, objects, rles, bboxs, scores):
        print(f"target_object: {target_object}")
        print(f"objects: {objects}")
        target_bbox = None
        target_rle = None
        max_score = -100
        class_name = None
        for i in range(len(objects)):
            if target_object in objects[i]:
                class_name = objects[i]
                if scores[i]>max_score:
                    target_bbox = bboxs[i]
                    target_rle = json.loads(rles[i])
        if target_rle is None:
            return None
        
        #get the mask from the compressed rle
        # get the image coords of the bounding box
        mask = mask_util.decode([target_rle])
        u_min = int(target_bbox[0])
        v_min = int(target_bbox[1])
        u_max = int(target_bbox[2])
        v_max = int(target_bbox[3])

        translation,rotation = self.listener.lookupTransform('world', self.cam_info.header.frame_id, rospy.Time(0))
        mat44 = self.listener.fromTranslationRotation(translation, rotation)

        #Some dumb logging to check rgb/depth alignment
        img_color = self.rgb_image.copy()
        cv2.imwrite("/home/rivr/color.png",img_color )
        if self.depth_encoding == "16UC1":
            img_depth = self.depth_image.copy()*1000
            img_depth = img_depth.astype('uint16')
        else:
            img_depth = self.depth_image.copy()
        cv2.imwrite("/home/rivr/depth.png",img_depth )
        imgray = mask*255
        cv2.imwrite("/home/rivr/mask.png",imgray )

        #get the camera center (cx,cy) and focal length (fx,fy)
        cx = self.cam_model.cx()
        cy = self.cam_model.cy()
        fx = self.cam_model.fx()
        fy = self.cam_model.fy()

        distances = []
        for v in range(v_min, v_max, 1):
            for u in range(u_min, u_max, 1):
                if mask[v][u] > 0:
                    d = (self.depth_image[v][u])
                    #if depth is 16bit int the depth value is in mm
                    # convert to m
                    if self.depth_encoding == "16UC1":
                        d = d/1000.0
                    #toss out points that are too close or too far 
                    # to simplify outlier rejection
                    if d <= 0.02 or d > 1.5 or np.isnan(d):
                        continue
                    distances.append((d, v, u))

        #Filter out outliers
        if len(distances)>0:
            distances = reject_outliers(np.asarray(distances), m=2)
            print(f"{class_name} num points: {len(distances)}")
        else:
            print(print(f"{class_name} no points found"))

        x = 0
        y = 0
        z = 0
        points = []
        #transform all the points into the world frame
        # and get the center
        for dist in distances:
            d = dist[0]
            v = dist[1]
            u = dist[2]
            ps = PointStamped()
            ps.header.frame_id = self.cam_info.header.frame_id
            ps.point.x = (u - cx)*d/fx
            ps.point.y = (v - cy)*d/fy
            ps.point.z = d
            t_p = transform_point(ps, mat44, 'world')

            if t_p.point.z > 0.025:
                x += t_p.point.x
                y += t_p.point.y
                z += t_p.point.z
                
                if class_name in self.colors:
                    b = self.colors[class_name][0]
                    g = self.colors[class_name][1]
                    r = self.colors[class_name][2]
                else:
                    b = 255
                    g = 255
                    r = 255

                points.append([t_p.point.x, t_p.point.y, t_p.point.z, r, g, b])

        num_points = len(points)
        cloud_msg = create_cloud(points, 'world')
        self.test_cloud.publish(cloud_msg)
        if num_points > 0:
            center = PointStamped()
            center.header.frame_id = 'world'
            center.point.x = x/num_points
            center.point.y = y/num_points
            center.point.z = z/num_points

            print(f"center: x:{center.point.x:.2f}, y:{center.point.y:.2f}, z:{center.point.z:.2f} in {center.header.frame_id}")        

            return center, cloud_msg
        else:
            return None
    
    def get_location_offset(self, target_object, objects, rles, bboxs, scores, text):
        print(f"target_object: {target_object}")
        print(f"objects: {objects}")
        target_bbox = None
        target_rle = None
        max_score = -100
        class_name = None
        for i in range(len(objects)):
            if target_object in objects[i]:
                class_name = objects[i]
                if scores[i]>max_score:
                    target_bbox = bboxs[i]
                    target_rle = json.loads(rles[i])
        if target_rle is None:
            return None
        
        req = ObjectLocationRequest()
        req.rgb_image = self.cvbridge.cv2_to_imgmsg(self.rgb_image)
        req.depth_image = self.cvbridge.cv2_to_imgmsg(self.depth_image)
        req.cam_info = self.cam_info
        req.object.class_name=target_object
        req.object.rle_encoded_mask= json.dumps(target_rle)
        req.object.u_min=target_bbox[0]
        req.object.v_min=target_bbox[1]
        req.object.u_max=target_bbox[2]
        req.object.v_max=target_bbox[3]
        req.transcription = text

        resp = self.part_location(req)

        print(f"main offset resp:\n{resp}\n-----")

        

        return resp



if __name__ == '__main__':
    llm = AssemblyClient()