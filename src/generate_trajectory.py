#!/usr/bin/env python3

import rospy
import message_filters
import math
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_inverse, quaternion_multiply

def get_position_error(p1, p2):
    x = p1.pose.position.x - p2.pose.position.x
    y = p1.pose.position.y - p2.pose.position.y
    z = p1.pose.position.z - p2.pose.position.z

    return (x,y,z)

def angle_axis(q):
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]

    sqrt_q = math.sqrt( qx**2 + qy**2 + qz**2 )

    ax = qx/sqrt_q
    ay = qy/sqrt_q
    az = qz/sqrt_q

    theta = math.atan2(sqrt_q, qw)

    return theta, ax, ay, az

def quat_from_orientation(orientation):
    q = [
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    ]

    return q

def distance(a, b):
    x = a[0]-b[0]
    y = a[1]-b[1]
    z = a[2]-b[2]

    return math.sqrt(x*x + y*y + z*z)

def quat_distance(q1,q2):
    x = q1[0]*q2[0]
    y = q1[1]*q2[1]
    z = q1[2]*q2[2]
    w = q1[3]*q2[3]
    
    return 1 - (x + y + z + w)**2

class GenerateTrajectory:
    def __init__(self):    
            rospy.init_node('generate_trajectory')
            self.twist_topic  = rospy.get_param("/twist_topic", "/my_gen3/servo_server/delta_twist_cmds")

            self.cart_vel_pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=10)
                
            self.front_slot_sub = message_filters.Subscriber("/front_slot", PoseStamped)
            self.back_slot_sub = message_filters.Subscriber("back_slot", PoseStamped)
            self.front_part_sub = message_filters.Subscriber("/front_part", PoseStamped)
            self.back_part_sub = message_filters.Subscriber("/back_part", PoseStamped)

            self.ts = message_filters.TimeSynchronizer([self.front_slot_sub, self.back_slot_sub, self.front_part_sub, self.back_part_sub], 10)
            self.ts.registerCallback(self.callback)
            rospy.spin()

    def callback(self, front_slot_pose, back_slot_pose, front_part_pose, back_part_pose):
        a = [
              front_slot_pose.pose.position.x,
              front_slot_pose.pose.position.y,
              front_slot_pose.pose.position.z
        ]
        b = [
              back_slot_pose.pose.position.x,
              back_slot_pose.pose.position.y,
              back_slot_pose.pose.position.z
        ]
        slot_distance = distance(a,b)
        a = [
              front_part_pose.pose.position.x,
              front_part_pose.pose.position.y,
              front_part_pose.pose.position.z
        ]
        b = [
              back_part_pose.pose.position.x,
              back_part_pose.pose.position.y,
              back_part_pose.pose.position.z
        ]
        part_distance = distance(a,b)
        q1 = [
              front_part_pose.pose.orientation.x,
              front_part_pose.pose.orientation.y,
              front_part_pose.pose.orientation.z,
              front_part_pose.pose.orientation.w
        ]
        q2 = [
              back_part_pose.pose.orientation.x,
              back_part_pose.pose.orientation.y,
              back_part_pose.pose.orientation.z,
              back_part_pose.pose.orientation.w
        ]
        part_quat_distance = quat_distance(q1,q2)
        
        #rospy.loginfo(f"slot_distance:{slot_distance:.3f}")
        #rospy.loginfo(f"part_distance:{part_distance:.3f}")
        #rospy.loginfo(f"part_quat_distance:{part_quat_distance}")
        
        #front
        #takes in a stampedpose
        positional_error_front = get_position_error(front_slot_pose, front_part_pose)
        q_target_front = quat_from_orientation(front_part_pose.pose.orientation)
        q_robot_front = quat_from_orientation(front_slot_pose.pose.orientation)
        q_r_front = quaternion_multiply( q_target_front , quaternion_inverse(q_robot_front))                
        angular_error_front, ax_front, ay_front, az_front = angle_axis(q_r_front)
        
        #back
        #takes in a stampedpose
        positional_error_back = get_position_error(back_slot_pose, back_part_pose)
        q_target_back = quat_from_orientation(back_part_pose.pose.orientation)
        q_robot_back = quat_from_orientation(back_slot_pose.pose.orientation)
        q_r_back = quaternion_multiply( q_target_back , quaternion_inverse(q_robot_back))                
        angular_error_back, ax_back, ay_back, az_back = angle_axis(q_r_back)

        '''
        rospy.loginfo(f"positional_error_front:{positional_error_front}")
        rospy.loginfo(f"angular_error_front:{angular_error_front}")
        rospy.loginfo(f"positional_error_back:{positional_error_back}")
        rospy.loginfo(f"angular_error_back:{angular_error_back}")
        '''
        
        t_l_x_front = positional_error_front[0]
        t_l_y_front = positional_error_front[1]
        t_l_z_front = positional_error_front[2]
        t_a_x_front = angular_error_front*ax_front
        t_a_y_front = angular_error_front*ay_front
        t_a_z_front = angular_error_front*az_front
        rospy.loginfo(f"[{t_l_x_front:.3f},{t_l_y_front:.3f},{t_l_z_front:.3f}],[{t_a_x_front:.3f},{t_a_y_front:.3f},{t_a_z_front:.3f}]")

        t_l_x_back = positional_error_back[0]
        t_l_y_back = positional_error_back[1]
        t_l_z_back = positional_error_back[2]
        t_a_x_back = angular_error_back*ax_back
        t_a_y_back = angular_error_back*ay_back
        t_a_z_back = angular_error_back*az_back
        rospy.loginfo(f"[{t_l_x_back:.3f},{t_l_y_back:.3f},{t_l_z_back:.3f}], [{t_a_x_back:.3f},{t_a_y_back:.3f},{t_a_z_back:.3f}]")
        rospy.loginfo("------------------------------")
        rospy.loginfo(f"[{(t_l_x_front-t_l_x_back):.3f},{(t_l_y_front-t_l_y_back):.3f},{(t_l_z_front-t_l_z_back):.3f}], [{(t_a_x_front-t_a_x_back):.3f},{(t_a_y_front-t_a_y_back):.3f},{(t_a_z_front-t_a_z_back):.3f}]")
        rospy.loginfo("------------------------------")


        #legs are not aligned -> part_quat_distance !~= 0
        #legs to far apart

        #legs to close together
        #legs out of workspace




        
if __name__ == '__main__':
    move = GenerateTrajectory()