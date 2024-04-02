#!/usr/bin/env python3

import rospy
import message_filters
import math
import numpy as np
from geometry_msgs.msg import PoseArray, TwistStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_inverse, quaternion_multiply
from toy_assembly.srv import Servo, ServoResponse

import moveit_commander
from simple_pid import PID

def zero_twist():
    zero_vel = TwistStamped()
    zero_vel.twist.linear.x = 0.0
    zero_vel.twist.linear.y = 0.0
    zero_vel.twist.linear.z = 0.0
    zero_vel.twist.angular.x = 0.0
    zero_vel.twist.angular.y = 0.0
    zero_vel.twist.angular.z = 0.0

    return zero_vel

def get_position_error(p1, p2):
    x = p1.position.x - p2.position.x
    y = p1.position.y - p2.position.y
    z = p1.position.z - p2.position.z

    return (x,y,z)

def angle_axis(q):
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]

    theta = 2*math.acos(qw)
    theta = theta % 2*math.pi
    s = math.sqrt(1-qw*qw)

    if (s < 0.001):
        ax=qx
        ay=qy
        az=qz
    else:
        ax=qx/s
        ay=qy/s
        az=qz/s

    '''
    sqrt_q = math.sqrt( qx**2 + qy**2 + qz**2 )

    ax = qx/sqrt_q
    ay = qy/sqrt_q
    az = qz/sqrt_q

    theta = math.atan2(sqrt_q, qw)
    '''

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

            joint_state_topic = ['joint_states:=/my_gen3_right/joint_states']

            moveit_commander.roscpp_initialize(joint_state_topic)

            self.robot = moveit_commander.RobotCommander(robot_description = "/my_gen3_right/robot_description")
            self.scene = moveit_commander.PlanningSceneInterface(ns="/my_gen3_right")

            rospy.loginfo("arm")
            self.arm_group_name = "arm"
            self.arm_move_group = moveit_commander.MoveGroupCommander(self.arm_group_name, ns="/my_gen3_right", robot_description = "/my_gen3_right/robot_description")
            self.arm_move_group.set_max_velocity_scaling_factor(1.0)




            self.twist_topic  = rospy.get_param("/twist_topic", "/my_gen3_right/servo/delta_twist_cmds")
            self.cart_vel_pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=10)
            rospy.loginfo(self.twist_topic)
            
            self.time_out = 15.0
            self.num_halt_msgs = 20
            self.pub_rate = 10
            self.t_old = rospy.Time.now()

            self.positional_tolerance = 0.03
            self.angular_tolerance = 0.3
                
            #proportional gains  
            self.cart_x_kp = rospy.get_param("~cart_x_kp", 10.50)
            self.cart_y_kp = rospy.get_param("~cart_y_kp", 10.50)
            self.cart_z_kp = rospy.get_param("~cart_z_kp", 10.50)
            self.angular_kp = rospy.get_param("~angular_kp", 1.50)

            #integral gains
            self.cart_x_ki = rospy.get_param("~cart_x_ki", 0.0)
            self.cart_y_ki = rospy.get_param("~cart_y_ki", 0.0)
            self.cart_z_ki = rospy.get_param("~cart_z_ki", 0.0)
            self.angular_ki = rospy.get_param("~angular_ki", 0.0)

            #derivative gains
            self.cart_x_kd = rospy.get_param("~cart_x_kd", 0.01)
            self.cart_y_kd = rospy.get_param("~cart_y_kd", 0.01)
            self.cart_z_kd = rospy.get_param("~cart_z_kd", 0.01)
            self.angular_kd = rospy.get_param("~angular_kd", 0.01)


            self.human_slot_sub = message_filters.Subscriber("/human_slot_array", PoseArray)
            self.robot_slot_sub = message_filters.Subscriber("/robot_slot_array", PoseArray)

            self.ts = message_filters.TimeSynchronizer([self.human_slot_sub, self.robot_slot_sub], 100)#, slop=12.0)
            self.ts.registerCallback(self.callback)


            self.service = rospy.Service('Servo', Servo, self.servo)


            rospy.spin()
            
    def callback(self, human_slot, robot_slot):
        #rospy.loginfo(len(human_slot.poses))
        #rospy.loginfo(len(robot_slot.poses))
       
        human_slot_pose = human_slot.poses[0]
        robot_slot_pose = robot_slot.poses[0]

        
        self.positional_error = get_position_error(robot_slot_pose, human_slot_pose)
        q_target = quat_from_orientation(human_slot_pose.orientation)
        q_robot = quat_from_orientation(robot_slot_pose.orientation)

        '''
        q_target = euler_from_quaternion(q_target)
        q_robot = euler_from_quaternion(q_robot)

        q_target = quaternion_from_euler(q_target[0],q_target[1],q_target[2])
        q_robot = quaternion_from_euler(q_robot[0],q_robot[1],q_robot[2])
        '''

        #print(f"q_rob:{q_robot}")
        #print(f"q_tar:{q_target}")

        self.q_r = quaternion_multiply(q_robot  , quaternion_inverse(q_target))

        #print(f"q_rot:{self.q_r}")

        self.angular_error, self.ax, self.ay, self.az = angle_axis(self.q_r)

        ee_pose = self.arm_move_group.get_current_pose()
        q_ee = np.asarray([
            ee_pose.pose.orientation.x,
            ee_pose.pose.orientation.y,
            ee_pose.pose.orientation.z,
            ee_pose.pose.orientation.w
        ])
        #print(f"q_ee:{ (180/math.pi)*np.asarray( euler_from_quaternion(q_ee))}")
        #print(f"q_ee:{ q_ee}")

        #np.set_printoptions(precision=2)
        #print(f"pe:  {np.asarray(self.positional_error)}")
        #print(f"q_r :{q_r}")


    def servo(self, req):
        x_pid = PID(Kp=self.cart_x_kp, Ki=self.cart_x_ki, Kd=self.cart_x_kd)
        y_pid = PID(Kp=self.cart_y_kp, Ki=self.cart_y_ki, Kd=self.cart_y_kd)
        z_pid = PID(Kp=self.cart_z_kp, Ki=self.cart_z_ki, Kd=self.cart_z_kd)
        x_pid.output_limits = (-1, 1)
        y_pid.output_limits = (-1, 1) 
        z_pid.output_limits = (-1, 1) 

        theta_pid = PID(Kp=self.angular_kp, Ki=self.angular_ki, Kd=self.angular_kd)
        theta_pid.output_limits = (-0.35, 0.35) 

        self.timed_out = False

        total_time = 0.0
        rate = rospy.Rate(self.pub_rate) # 10hz
        dt = 1.0/self.pub_rate
        while (not self.satisfy_tolerance(self.angular_error, self.positional_error) and total_time < self.time_out): 
            
            #get twist linear values from PID controllers
            
            t_l_x = x_pid(self.positional_error[0], dt)
            t_l_y = y_pid(self.positional_error[1], dt)
            t_l_z = z_pid(self.positional_error[2], dt)

            #get twist angular values
            #   get euler angles from axis angles of quaternion
            ang_vel_magnitude = theta_pid(self.angular_error, dt)
            t_a_x = ang_vel_magnitude * self.ax
            t_a_y = ang_vel_magnitude * self.ay
            t_a_z = ang_vel_magnitude * self.az

            '''
            t_l_x = self.positional_error[0]
            t_l_y = self.positional_error[1]
            t_l_z = self.positional_error[2]
            t_a_x = self.angular_error*self.ax
            t_a_y = self.angular_error*self.ay
            t_a_z = self.angular_error*self.az
            '''
            
            rospy.loginfo(f"total time:{total_time}\n[{t_l_x:.3f},{t_l_y:.3f},{t_l_z:.3f}]\n[{t_a_x:.3f},{t_a_y:.3f},{t_a_z:.3f}]")
            print(f"[{self.positional_error}], {self.positional_tolerance}")
            print(f"{self.angular_error}, {self.angular_tolerance}")
            #print(f"[{t_l_x:.3f},{t_l_y:.3f},{t_l_z:.3f}]")
            #print(f"[{t_a_x:.3f},{t_a_y:.3f},{t_a_z:.3f}]")


            twist = TwistStamped()
            twist.header.stamp = rospy.Time.now()
            twist.header.frame_id = "right_base_link"
            twist.twist.linear.x = t_l_x if abs(t_l_x) > self.positional_tolerance else 0.0
            twist.twist.linear.y = t_l_y if abs(t_l_y) > self.positional_tolerance else 0.0
            twist.twist.linear.z = t_l_z if abs(t_l_z) > self.positional_tolerance else 0.0
            twist.twist.angular.x = t_a_x if abs(t_a_x) > self.angular_tolerance else 0.0
            twist.twist.angular.y = t_a_y if abs(t_a_y) > self.angular_tolerance else 0.0
            twist.twist.angular.z = t_a_z if abs(t_a_z) > self.angular_tolerance else 0.0

            if abs(t_l_x) > self.positional_tolerance and abs(t_l_y) > self.positional_tolerance:
                twist.twist.angular.z = 0.0


            #rospy.loginfo(twist)
            self.cart_vel_pub.publish(twist)        #send zero twist to halt servoing

            total_time += dt
            if total_time > self.time_out:
                self.timed_out = True

            rate.sleep()
            

        if  self.timed_out:
            rospy.loginfo("Servoing timed out")
        else:
            rospy.loginfo("Servoing took %f seconds" % total_time)

        #sehnd out series of zero twist to halt servoing
        pose_vel = zero_twist()
        pose_vel.header.frame_id = "right_base_link"
        rate = rospy.Rate(self.pub_rate) # 10hz
        for i in range(self.num_halt_msgs):
            pose_vel.header.stamp = rospy.Time.now()
            self.cart_vel_pub.publish(pose_vel)
            rate.sleep()


        return ServoResponse(not self.timed_out)

    def satisfy_tolerance(self, angular_error, positional_error):
        x_err = positional_error[0]
        y_err = positional_error[1]
        z_err = positional_error[2]

        return (abs(x_err) < self.positional_tolerance and
                abs(y_err) < self.positional_tolerance and
                abs(z_err) < self.positional_tolerance and
                abs(angular_error) < self.angular_tolerance)
        
if __name__ == '__main__':
    move = GenerateTrajectory()