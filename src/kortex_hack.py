#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import TwistStamped
from kortex_driver.msg import TwistCommand
from kortex_driver.msg import Base_JointSpeeds, JointSpeed
from trajectory_msgs.msg import JointTrajectory

def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

class KortexHack:
    def __init__(self):
        rospy.init_node('kortex_fix', anonymous=True)

        self.arm = rospy.get_param("~prefix", default="right")
        rospy.loginfo(self.arm)

        self.servo_sub = rospy.Subscriber('/my_gen3_'+self.arm+'/servo/delta_twist_cmds', TwistStamped, self.delta_twist_cmds__to_cart_cb)
        self.cart_vel_pub = rospy.Publisher('/my_gen3_'+self.arm+'/in/cartesian_velocity', TwistCommand, queue_size=10)

        #self.servo_sub = rospy.Subscriber('/my_gen3_'+self.arm+'/'+self.arm+'_gen3_joint_trajectory_controller/command', JointTrajectory, self.joint_command_cb)
        #self.joint_vel_pub = rospy.Publisher('/my_gen3_'+self.arm+'/in/joint_velocity', Base_JointSpeeds, queue_size=10)

        self.min_linear_vel = -0.1
        self.max_linear_vel =  0.1
        self.min_angular_vel = -0.2
        self.max_angular_vel =  0.2

        rospy.spin()


    def joint_command_cb(self, joint_command):
        joint_speeds = Base_JointSpeeds()
        duration = joint_command.points[0].time_from_start.to_nsec()
        joint_speeds.duration=duration

        for j in range(len(joint_command.joint_names)):
            #rospy.loginfo(f"{joint_command.joint_names[j]}, {joint_command.points[0].velocities[j]}")
            joint_vel = JointSpeed()
            joint_vel.joint_identifier = j
            joint_vel.value = joint_command.points[0].velocities[j]
            joint_vel.duration = duration
            joint_speeds.joint_speeds.append(joint_vel)

        self.joint_vel_pub.publish(joint_speeds)


    def delta_twist_cmds__to_cart_cb(self, delta_twist):
        twist = TwistCommand()
        twist.reference_frame = 1 #2 tool frame (see https://github.com/Kinovarobotics/ros_kortex/blob/bf71d8ecdec874a88d02b1a2481784ae2a7fdba4/kortex_driver/protos/Common.proto#L207)
        twist.duration = 0

        twist.twist.linear_x = clamp(delta_twist.twist.linear.x, self.min_linear_vel, self.max_linear_vel)
        twist.twist.linear_y = clamp(delta_twist.twist.linear.y, self.min_linear_vel, self.max_linear_vel)
        twist.twist.linear_z = clamp(delta_twist.twist.linear.z, self.min_linear_vel, self.max_linear_vel)

        twist.twist.angular_x = clamp(delta_twist.twist.angular.x, self.min_angular_vel, self.max_angular_vel)
        twist.twist.angular_y = clamp(delta_twist.twist.angular.y, self.min_angular_vel, self.max_angular_vel)
        twist.twist.angular_z = clamp(delta_twist.twist.angular.z, self.min_angular_vel, self.max_angular_vel)

        self.cart_vel_pub.publish(twist)

if __name__ == '__main__':
    physical_arm = KortexHack()
