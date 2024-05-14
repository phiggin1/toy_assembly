#!/usr/bin/env python3

import pygame
import rospy
import tf
import cv2
import math
import json
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped, PoseStamped
from toy_assembly.msg import ObjectImage
from toy_assembly.srv import MoveITPose, MoveITPoseRequest
from toy_assembly.srv import OrientCamera
import numpy as np
from multiprocessing import Lock

from std_srvs.srv import Trigger 

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_COLOR = (255,255,255)
FONT_THICKNESS = 1
LINE_TYPE = 1

def get_button_image(img, text):
    if len(text) > len("release"):
        split_text = text[5:].split("_")
        wrapped_text = [
            "_".join(split_text[0:2]),
            "_".join(split_text[2:4])
        ]
    else:
        wrapped_text = [text]

    x = 0
    y = 5

    for i, line in enumerate(wrapped_text):
        #print(line)
        textsize = cv2.getTextSize(line, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        gap = textsize[1]
        y = 10 + int(textsize[1]/2) + (i*gap)
        cv2.putText(img, line, (x, y), FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)

    return img

class ManualServo:
    def __init__(self):    
        rospy.init_node('manual_servo', anonymous=True)
        self.cvbridge = CvBridge()

        self.mutex = Lock()

        self.right_planning_frame = 'right_base_link'
        self.listener = tf.TransformListener()
        pygame.init()
        width = 640
        height = 480
        self.shape = (width+10,height+10)
        print(self.shape)


        self.real = rospy.get_param("~real", default="false")
        self.arm_prefix = rospy.get_param("~arm_prefix", default="right")
        rospy.loginfo(f"arm: {self.arm_prefix}")
        rospy.loginfo(f"real: {self.real}")
        if self.real:
            rgb_topic = "/right_camera/color/image_raw"
        else:
            rgb_topic = "/unity/camera/right/rgb/image_raw"
        rospy.loginfo(f"rgb_topic: {rgb_topic}")


        if self.real:
            object_image_topic = "/left_camera/color/image_raw"
            #object_image_topic = "/object_images"
        else:
            object_image_topic = "/unity/camera/left/rgb/image_raw"
            #object_image_topic = "/object_images"
        rospy.loginfo(f"object_image_topic: {object_image_topic}")


        self.rgb_img = None
        self.rgb_image_sub = rospy.Subscriber(rgb_topic, Image, self.image_cb)

        self.object_img = None
        #self.object_image_sub = rospy.Subscriber(object_image_topic, ObjectImage, self.object_image_cb)
        self.object_image_sub = rospy.Subscriber(object_image_topic, Image, self.object_image_cb)


        self.pgscreen=pygame.display.set_mode((self.shape[0]*2+50,self.shape[1]+5+30))
        self.pgscreen.fill((255, 255, 255))
        pygame.display.set_caption('ManualServo')     

        self.camera_names = [
                            ['hand_pointing_right_cam_up', [math.sqrt(2)/2, 0, 0, math.sqrt(2)/2]],
                            ['hand_pointing_right_cam_front', [0.5, 0.5, -0.5, 0.5]],
                            ['hand_pointing_left_cam_up', [0, math.sqrt(2)/2, math.sqrt(2)/2, 0]],
                            ['hand_pointing_left_cam_front', [-0.5, -0.5, -0.5, -0.5]],
                            ['hand_pointing_down_cam_right', [-1, 0, 0, 0]],
                            ['hand_pointing_down_cam_front', [-math.sqrt(2)/2, -math.sqrt(2)/2, 0, 0]],
                            ['hand_pointing_forward_cam_up', [0.5, 0.5, 0.5, 0.5]],
                            ['hand_pointing_forward_cam_right', [math.sqrt(2)/2, 0, math.sqrt(2)/2, 0]],
                            ['grab'],
                            ['release']
        ]
      
        self.camera_buttons = pygame.sprite.Group()
        self.object_buttons = pygame.sprite.Group()
        if self.arm_prefix == "right":
            y = self.shape[1]+15
            x = 5
            place = 65
            for name in self.camera_names:
                button = pygame.sprite.Sprite()
                button.name = name[0]
                blank_image = get_button_image(np.zeros((30,120,3), np.uint8), name[0])
                button.image = pygame.image.frombuffer(blank_image.tobytes(), blank_image.shape[1::-1], "RGB")
                button.rect = button.image.get_rect()
                button.rect.center = (place, y)
                place += 125
                self.camera_buttons.add(button)
            self.camera_buttons.draw(self.pgscreen)
            '''
            self.object_names = range(5)
            y = 15
            x = self.shape[0]*2+10
            place = 35
            for name in self.object_names:
                button = pygame.sprite.Sprite()
                button.name = name
                blank_image = get_button_image(np.zeros((23,25,3), np.uint8), str(name))
                button.image = pygame.image.frombuffer(blank_image.tobytes(), blank_image.shape[1::-1], "RGB")
                button.rect = button.image.get_rect()
                button.rect.center = (x, y)
                y += place
                self.object_buttons.add(button)
            self.object_buttons.draw(self.pgscreen)
            '''        

        self.angular_vel = 0.1
        self.linear_vel = 0.2

        self.twist_topic  = rospy.get_param("/twist_topic", "/my_gen3_"+self.arm_prefix+"/servo/delta_twist_cmds")
        self.cart_vel_pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=10)
        rospy.loginfo(self.twist_topic)

        self.button_topic  = rospy.get_param("/button_topic", "/buttons")
        self.button_pub = rospy.Publisher(self.button_topic, String, queue_size=10)
        rospy.loginfo(self.button_topic)

        while self.rgb_img is None :
            with self.mutex:
                rospy.sleep(.1)
        while self.object_img is None:
            with self.mutex:
                rospy.sleep(.1)

        self.orientation =  [-math.sqrt(2)/2, -math.sqrt(2)/2, 0, 0]


    def run(self):
        rate = rospy.Rate(30)
        self.zeros = 0
        running = True
        while not rospy.is_shutdown() and running:

            self.get_input()
            rate.sleep()

    def get_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rospy.loginfo("pygame.QUIT")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                for i, button in enumerate(self.camera_buttons):
                    if (button.rect.left < pos[0] < button.rect.right) and (button.rect.top < pos[1] < button.rect.bottom):
                        print(button.name, self.camera_names[i], i)
                        print(len(self.camera_buttons))
                        if (i < (len(self.camera_buttons)-2)):
                            self.orient_camera(self.camera_names[i][0])
                            self.orientation = self.camera_names[i][1]
                        elif (button.name == "grab"):
                            self.grab()
                        elif (button.name == "release"):
                            self.release()
                for i, button in enumerate(self.object_buttons):
                    if (button.rect.left < pos[0] < button.rect.right) and (button.rect.top < pos[1] < button.rect.bottom):
                        print(button.name, self.object_names[i])
                        if button.name < len(self.object_positions):
                            print(f"{button.name}, {self.object_positions[button.name]}")
                            self.move_to_position(self.object_positions[button.name])
                            

        '''
        Get the keys pressed for moving the end effector around
        in the frame of the end effector
        q/e up/down in ee space
        w/s forward back in ee space
        a/d left right in ee space

        num2/8 (x)pitch in ee space < double check
        num4/6 (z)roll in ee space < double check
        num7/9 (y)yaw in ee space < double check
        '''
        cmd = TwistStamped()
        cmd.header.frame_id = self.arm_prefix+"_end_effector_link"
        key = None
        keys = pygame.key.get_pressed()
        pressed = False
        #if a key is pressed reset the zero twist message counter to 0
        if keys[pygame.K_w]:
            key = pygame.K_w
            pressed = True
            self.zeros = 0
            cmd.twist.linear.z = self.linear_vel
        elif keys[pygame.K_s]:
            key = pygame.K_s
            pressed = True
            self.zeros = 0
            cmd.twist.linear.z = -self.linear_vel

        if keys[pygame.K_a]:
            key = pygame.K_a
            pressed = True
            self.zeros = 0
            cmd.twist.linear.x = self.linear_vel
        elif keys[pygame.K_d]:
            key = pygame.K_d
            pressed = True
            self.zeros = 0
            cmd.twist.linear.x = -self.linear_vel
        
        if keys[pygame.K_q]:
            key = pygame.K_q
            pressed = True
            self.zeros = 0
            cmd.twist.linear.y = self.linear_vel
        elif keys[pygame.K_e]:
            key = pygame.K_e
            pressed = True
            self.zeros = 0
            cmd.twist.linear.y = -self.linear_vel


        #pitch
        if keys[pygame.K_i]:
            key = pygame.K_i
            pressed = True
            self.zeros = 0
            cmd.twist.angular.x = self.angular_vel
        elif keys[pygame.K_k]:
            key = pygame.K_k
            pressed = True
            self.zeros = 0
            cmd.twist.angular.x = -self.angular_vel
        
        #yaw
        if keys[pygame.K_j]:
            key = pygame.K_j
            pressed = True
            self.zeros = 0
            cmd.twist.angular.y = self.angular_vel
        elif keys[pygame.K_l]:
            key = pygame.K_l
            pressed = True
            self.zeros = 0
            cmd.twist.angular.y = -self.angular_vel
        
        #roll
        if keys[pygame.K_u]:
            key = pygame.K_u
            pressed = True
            self.zeros = 0
            cmd.twist.angular.z = -self.angular_vel
        elif keys[pygame.K_o]:
            key = pygame.K_o
            pressed = True
            self.zeros = 0
            cmd.twist.angular.z = self.angular_vel

        if not pressed:
            self.zeros += 1

        rospy.loginfo(f"{key}, {self.zeros}")
        if key is not None:
            print(f"\tl_x:{cmd.twist.linear.x}, l_y:{cmd.twist.linear.y}, l_z:{cmd.twist.linear.z}")
            print(f"\ta_x:{cmd.twist.angular.x}, a_y:{cmd.twist.angular.y}, a_z:{cmd.twist.angular.z}")
        
        #only send out messages if fewer than 10 zero twist messages have been sent to allow other nodes to use moveit
        # or if a command is given
        if pressed or self.zeros < 10:
            cmd.header.stamp = rospy.Time.now()
            self.cart_vel_pub.publish(cmd)

    def image_cb(self, rgb):
        with self.mutex:
            self.rgb_img = self.cvbridge.imgmsg_to_cv2(rgb, "bgr8")
            self.rgb_img = cv2.resize(self.rgb_img, (640, 480))
            pg_img = pygame.image.frombuffer(cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2RGB).tobytes(), self.rgb_img.shape[1::-1], "RGB")
            self.pgscreen.blit(pg_img, (5,5))
            pygame.display.update()

    def object_image_cb(self, object_image):
        with self.mutex:
            self.header = object_image.header
            '''
            self.object_positions = object_image.object_positions
            self.object_img = self.cvbridge.imgmsg_to_cv2(object_image.image, "bgr8")
            '''
            self.object_img = self.cvbridge.imgmsg_to_cv2(object_image, "bgr8")

            self.object_img = cv2.resize(self.object_img, (640, 480))
            pg_img = pygame.image.frombuffer(cv2.cvtColor(self.object_img, cv2.COLOR_BGR2RGB).tobytes(), self.object_img.shape[1::-1], "RGB")
            self.pgscreen.blit(pg_img, (self.shape[0],5))
            pygame.display.update()
    
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

        print("release")
        a = dict()
        a["robot"] = self.arm_prefix
        a["action"] = "released"
        s = json.dumps(a)
        self.button_pub.publish(s)

    def orient_camera(self, pose_str):
        print(pose_str)
        pose_name = String()
        pose_name.data = pose_str
        service_name = "/my_gen3_right/rotate_object"
        rospy.wait_for_service(service_name)
        print(service_name)
        try:
            orient_camera = rospy.ServiceProxy(service_name, OrientCamera)
            resp = orient_camera(pose_str)
            return resp
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s"%e)

    def move_to_position(self, position):
        pose = PoseStamped()
        pose.header = self.header
        pose.pose.position = position
        pose.pose.orientation.w = 1.0
        pose = self.transform_obj_pose(pose)
        pose.pose.position.z += 0.2

        pose.pose.orientation.x = self.orientation[0]
        pose.pose.orientation.y = self.orientation[1]
        pose.pose.orientation.z = self.orientation[2]
        pose.pose.orientation.w = self.orientation[3]
        print(pose)
        rospy.wait_for_service('/my_gen3_right/move_pose')
        print("'/my_gen3_right/move_pose'")
        try:
            moveit_pose = rospy.ServiceProxy('/my_gen3_right/move_pose', MoveITPose)
            resp = moveit_pose(pose)
            return resp
        except rospy.ServiceException as e:
            rospy.loginfo("Service call failed: %s"%e)

    def transform_obj_pose(self, obj_pose):
        t = rospy.Time.now()
        obj_pose.header.stamp = t
        self.listener.waitForTransform(obj_pose.header.frame_id, self.right_planning_frame, t, rospy.Duration(4.0))
        obj_pose = self.listener.transformPose(self.right_planning_frame, obj_pose)
        return obj_pose

if __name__ == '__main__':
    move = ManualServo()
    move.run()

'''
pygame
Constant      ASCII   Description
---------------------------------
K_BACKSPACE   \b      backspace
K_TAB         \t      tab
K_CLEAR               clear
K_RETURN      \r      return
K_PAUSE               pause
K_ESCAPE      ^[      escape
K_SPACE               space
K_EXCLAIM     !       exclaim
K_QUOTEDBL    "       quotedbl
K_HASH        #       hash
K_DOLLAR      $       dollar
K_AMPERSAND   &       ampersand
K_QUOTE               quote
K_LEFTPAREN   (       left parenthesis
K_RIGHTPAREN  )       right parenthesis
K_ASTERISK    *       asterisk
K_PLUS        +       plus sign
K_COMMA       ,       comma
K_MINUS       -       minus sign
K_PERIOD      .       period
K_SLASH       /       forward slash
K_0           0       0
K_1           1       1
K_2           2       2
K_3           3       3
K_4           4       4
K_5           5       5
K_6           6       6
K_7           7       7
K_8           8       8
K_9           9       9
K_COLON       :       colon
K_SEMICOLON   ;       semicolon
K_LESS        <       less-than sign
K_EQUALS      =       equals sign
K_GREATER     >       greater-than sign
K_QUESTION    ?       question mark
K_AT          @       at
K_LEFTBRACKET [       left bracket
K_BACKSLASH   \       backslash
K_RIGHTBRACKET ]      right bracket
K_CARET       ^       caret
K_UNDERSCORE  _       underscore
K_BACKQUOTE   `       grave
K_a           a       a
K_b           b       b
K_c           c       c
K_d           d       d
K_e           e       e
K_f           f       f
K_g           g       g
K_h           h       h
K_i           i       i
K_j           j       j
K_k           k       k
K_l           l       l
K_m           m       m
K_n           n       n
K_o           o       o
K_p           p       p
K_q           q       q
K_r           r       r
K_s           s       s
K_t           t       t
K_u           u       u
K_v           v       v
K_w           w       w
K_x           x       x
K_y           y       y
K_z           z       z
K_DELETE              delete
K_KP0                 keypad 0
K_KP1                 keypad 1
K_KP2                 keypad 2
K_KP3                 keypad 3
K_KP4                 keypad 4
K_KP5                 keypad 5
K_KP6                 keypad 6
K_KP7                 keypad 7
K_KP8                 keypad 8
K_KP9                 keypad 9
K_KP_PERIOD   .       keypad period
K_KP_DIVIDE   /       keypad divide
K_KP_MULTIPLY *       keypad multiply
K_KP_MINUS    -       keypad minus
K_KP_PLUS     +       keypad plus
K_KP_ENTER    \r      keypad enter
K_KP_EQUALS   =       keypad equals
K_UP                  up arrow
K_DOWN                down arrow
K_RIGHT               right arrow
K_LEFT                left arrow
K_INSERT              insert
K_HOME                home
K_END                 end
K_PAGEUP              page up
K_PAGEDOWN            page down
K_F1                  F1
K_F2                  F2
K_F3                  F3
K_F4                  F4
K_F5                  F5
K_F6                  F6
K_F7                  F7
K_F8                  F8
K_F9                  F9
K_F10                 F10
K_F11                 F11
K_F12                 F12
K_F13                 F13
K_F14                 F14
K_F15                 F15
K_NUMLOCK             numlock
K_CAPSLOCK            capslock
K_SCROLLOCK           scrollock
K_RSHIFT              right shift
K_LSHIFT              left shift
K_RCTRL               right control
K_LCTRL               left control
K_RALT                right alt
K_LALT                left alt
K_RMETA               right meta
K_LMETA               left meta
K_LSUPER              left Windows key
K_RSUPER              right Windows key
K_MODE                mode shift
K_HELP                help
K_PRINT               print screen
K_SYSREQ              sysrq
K_BREAK               break
K_MENU                menu
K_POWER               power
K_EURO                Euro
K_AC_BACK             Android back button
'''