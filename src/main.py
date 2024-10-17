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

    text_json = text[a:b]

    try:
        json_dict = json.loads(text_json)
        return json_dict
    except:
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
        self.sim_time = rospy.get_param("/use_sime_time", False)

        rospy.loginfo(f"sim time active: {self.sim_time}")

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
        
        phi_service_name = "/phi_servcice"
        rospy.wait_for_service(phi_service_name)
        self.llm_text_srv = rospy.ServiceProxy(phi_service_name, LLMText)

        sam_service_name = "/get_sam_segmentation"
        rospy.wait_for_service(sam_service_name)
        self.sam_srv = rospy.ServiceProxy(sam_service_name, SAM)

        self.robot_speech_pub = rospy.Publisher('/text_to_speech', String, queue_size=10)
        self.debug_pose_pub = rospy.Publisher('/0debug_pose', PoseStamped, queue_size=10)
        self.status_pub = rospy.Publisher("/status", String, queue_size=10)

        self.twist_topic  = "/my_gen3_right/workspace/delta_twist_cmds"
        self.cart_vel_pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=10)
        
        self.test_cloud = rospy.Publisher("test_cloud", PointCloud2, queue_size=10)

        fp_init_env = "/home/rivr/toy_ws/src/toy_assembly/prompts/init_env.txt"
        with open(fp_init_env) as f:
            self.init_env = f.read()
        self.env = None

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
        
        rate = rospy.Rate(5)
        while not self.have_images:
            rate.sleep()
        

        rospy.on_shutdown(self.shutdown_hook)
        while not rospy.is_shutdown():
            '''
            transcription = rospy.wait_for_message("/transcript", Transcription)
            '''
            text = input("command: ")
            transcription = Transcription()
            transcription.transcription = text
            
            self.text_cb(transcription)

    def image_cb(self, rgb_ros_image, depth_ros_image):
        success = self.mutex.acquire(block=False, timeout=0.02)
        if success:
            #print(f"rgb eoncoding: {rgb_ros_image.encoding}")
            #print(f"depth eoncoding: {depth_ros_image.encoding}")
            self.rgb_image = self.cvbridge.imgmsg_to_cv2(rgb_ros_image, desired_encoding="bgr8") 
            self.depth_image = self.cvbridge.imgmsg_to_cv2(depth_ros_image)#, desired_encoding="passthrough") 
            if not self.have_images:
                rospy.loginfo("HAVE IMAGES")
                self.have_images = True
            self.mutex.release()

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


        print(results)
        print(self.env)

        '''
        check if env makes sense with what is seen
        '''

        rospy.loginfo(results)
        rospy.loginfo("WAITING")
        self.status_pub.publish("WAITING")
        if self.debug: rospy.loginfo("--------------------------------------------------------") 
        self.dataframe_csv.append(self.df)


    def high_level(self, text):
        rospy.loginfo("waiting for objects")

        image, objects, rles, bboxs, scores = self.get_detections("tan tray. orange tray. tan horse body. blue horse legs. orange horse legs. table.")

        req = LLMImageRequest()
        req.text = text
        if self.env is None:
            self.env = self.init_env
        req.env = self.env
        req.image = image

        resp = self.llm_image_srv(req)

        cv_img = self.cvbridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        merge_test = "_".join(text.split(" "))
        fname = f"/home/rivr/toy_logs/images/{image.header.stamp}{merge_test}.png"
        print(fname)
        cv2.imwrite(fname, cv_img)

        self.df["image_path"] = [fname]
        self.df["objects"] = [objects]
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
                if "object" in json_dict:
                    print(json_dict["object"])
                    target_object = json_dict["object"]

                    target_position = self.get_position(target_object, objects, rles, bboxs, scores)

                    success = self.pickup(target_position)
                    self.env = json.dumps(json_dict["environment_after"], indent=4)
                    results = ("PICKUP", success)
            else:
                results = ("PICKUP", False)
        elif "MOVE_TO" in action:
            if len(objects) > 0:
                if "object" in json_dict:
                    print(json_dict["object"])
                    target_object = json_dict["object"]

                    target_position = self.get_position(target_object, objects, rles, bboxs, scores)

                    print(f"{target_object}, {target_position.header.frame_id}, x:{target_position.point.x}, x:{target_position.point.y}, x:{target_position.point.z}")
                    success = self.move_to(target_position)
                    self.env = json.dumps(json_dict["environment_after"], indent=4)
                    results = ("MOVE_TO", success)
                else:
                    results = ("MOVE_TO", False)
            else:
                results = ("MOVE_TO", False)

        else:   
            any_valid_commands = self.ee_move(action)
            results = (action, any_valid_commands)
        '''
        check if predicted state and actual state match
        if not update self.env
        '''

        self.state = "LOW_LEVEL"
        return results

    def low_level(self, text):
        req = LLMTextRequest()
        req.text = text
        '''
        if self.env is None:
            self.env = self.init_env
        req.env = self.env
        '''
        resp = self.llm_text_srv(req)

        self.df["phi3_response"] = [str(resp.text).replace('\n','')]
        action = self.parse_llm_response(resp.text)

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

    def parse_llm_response(self, text):
        print(text)
        json_dict = extract_json(text)
        if json_dict is None:
            return None
        
        action = None
        if "action" in json_dict:
            action = json_dict["action"]

        return action


    def pickup(self, target_position):
        open_succes = self.open()
        rospy.loginfo(f"open_succes: {open_succes}")
        if not open_succes:
            return open_succes

        rospy.loginfo(f"pickup: {target_position.point.x},  {target_position.point.y}. {target_position.point.z}")
        init_grab_move_to_success = self.move_to(target_position)
        rospy.loginfo(f"init grab move_to successful : {init_grab_move_to_success}")
        if not init_grab_move_to_success:
            return init_grab_move_to_success

        grab_success = self.grab(target_position)
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

    def grab(self, position):
        rospy.loginfo(f"grab:{position.header.frame_id} {position.point.x:.3f}, {position.point.y:.3f}, {position.point.z:.3f}")
        final_pose = PoseStamped()
        final_pose.header = position.header
        final_pose.pose.position = deepcopy(position.point)
        final_pose.pose.orientation.x = -1
        final_pose.pose.orientation.w = 0
        final_pose.pose.position.z -= 0.03725

        min_safe_height = 0.065
        final_pose.pose.position.z = max(min_safe_height, final_pose.pose.position.z)

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
        
   
        mask = mask_util.decode([target_rle])
        u_min = int(target_bbox[0])
        v_min = int(target_bbox[1])
        u_max = int(target_bbox[2])
        v_max = int(target_bbox[3])

        translation,rotation = self.listener.lookupTransform('world', self.cam_info.header.frame_id, rospy.Time(0))
        mat44 = self.listener.fromTranslationRotation(translation, rotation)

        min_x = 1000.0
        min_y = 1000.0
        min_z = 1000.0
        max_x = -1000.0
        max_y = -1000.0
        max_z = -1000.0

        #get the camera center (cx,cy) and focal length (fx,fy)
        cx = self.cam_model.cx()
        cy = self.cam_model.cy()
        fx = self.cam_model.fx()
        fy = self.cam_model.fy()

        distances = []
        for v in range(v_min, v_max, 2):
            for u in range(u_min, u_max, 2):
                if mask[v][u] > 0:
                    d = (self.depth_image[v][u])
                    #if d > 100 depth is in mm
                    if d > 100:
                        d = d/1000.0
                    if d <= 0.001 or d > 1.0 or np.isnan(d):
                        continue
                    distances.append((d, v, u))

        if len(distances)>0:
            distances = reject_outliers(np.asarray(distances), m=1)
            print(f"{class_name} num points: {len(distances)}")
        else:
            print(print(f"{class_name} no points found"))

        points = []

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

            if t_p.point.x > max_x:
                max_x = t_p.point.x
            if t_p.point.x < min_x:
                min_x = t_p.point.x

            if t_p.point.y > max_y:
                max_y = t_p.point.y
            if t_p.point.y < min_y:
                min_y = t_p.point.y

            if t_p.point.z > max_z:
                max_z = t_p.point.z
            if t_p.point.z < min_z:
                min_z = t_p.point.z
            
            if class_name in self.colors:
                b = self.colors[class_name][0]
                g = self.colors[class_name][1]
                r = self.colors[class_name][2]
            else:
                b = 255
                g = 255
                r = 255

            points.append([t_p.point.x, t_p.point.y, t_p.point.z, r, g, b])

        self.test_cloud.publish(create_cloud(points, 'world'))

        center = PointStamped()
        center.header.frame_id = 'world'
        center.point.x = (min_x + max_x)/2
        center.point.y = (min_y + max_y)/2
        center.point.z = (min_z + max_z)/2

        print(f"center: x:{((min_x + max_x)/2):.2f}, y:{((min_y + max_y)/2):.2f}, z:{((min_z + max_z)/2):.2f}")        

        return center
    
if __name__ == '__main__':
    llm = AssemblyClient()