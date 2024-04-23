#!/usr/bin/env python3

import pygame
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped


class ManualServo:
    def __init__(self):    
            rospy.init_node('manual_servo')
            self.cvbridge = CvBridge()

            self.rgb_img = None
            self.rgb_image_sub = rospy.Subscriber("/unity/camera/right/rgb/image_raw", Image, self.image_cb)

            self.twist_topic  = rospy.get_param("/twist_topic", "/my_gen3_right/servo/delta_twist_cmds")
            self.cart_vel_pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=10)
            rospy.loginfo(self.twist_topic)


            while self.rgb_img is None:
                rospy.sleep(.1)

            shape = self.rgb_img.shape()

            pygame.init()
            pgscreen=pygame.display.set_mode(shape)
            pgscreen.fill((255, 255, 255))
            pygame.display.set_caption('ManualServo')
            rate = rospy.Rate(20)
            zeros = 0
            while not rospy.is_shutdown():
                
                pg_img = pygame.image.frombuffer(cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2RGB).tostring(), self.rgb_img.shape[1::-1], "RGB")
                pgscreen.blit(pg_img, (5,5))
                pygame.display.update()                
                
                cmd = TwistStamped()
                cmd.header.frame_id = "right_base_link"
                
                keys = pygame.key.get_pressed()
                rospy.loginfo("")
                pressed = False
                if keys[pygame.K_UP]:
                    pressed = True
                    zeros = 0
                    print('K_UP arrow key')
                    cmd.twist.linear.x = 1.0
                elif keys[pygame.K_DOWN]:
                    pressed = True
                    zeros = 0
                    print('K_DOWN arrow key')
                    cmd.twist.linear.x = -1.0
                elif keys[pygame.K_RIGHT]:
                    pressed = True
                    zeros = 0
                    print('K_RIGHT arrow key')
                    cmd.twist.angular.z = -1.0
                elif keys[pygame.K_LEFT]:
                    pressed = True
                    zeros = 0
                    print('K_LEFT arrow key')
                    cmd.twist.angular.z = 1.0
                else:
                    zeros += 1

                if pressed or zeros < 5:
                    cmd.header.stamp = rospy.Time.now()
                    self.cart_vel_pub.publish(cmd)
            
                rate.sleep()

    def image_cb(self, rgb):
        self.rgb_img = self.cvbridge.imgmsg_to_cv2(rgb, "bgr8")

if __name__ == '__main__':
    move = ManualServo()