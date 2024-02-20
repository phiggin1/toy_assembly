#!/usr/bin/env python3

import rospy
import json
import math
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

FRONT_SLOT_NAME="/horse_body_blue/FrontSlotTarget"
BACK_SLOT_NAME="/horse_body_blue/BackSlotTarget"

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
     
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
     
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
     
    return roll_x, pitch_y, yaw_z # in radians

def distance(a, b):
    x = a[0]-b[0]
    y = a[1]-b[1]
    z = a[2]-b[2]

    return math.sqrt(x*x + y*y + z*z)

def distance_point_to_ray(pA, pB, p):
    d = np.linalg.norm( np.cross( p-pA, p-pB ) ) / np.linalg.norm( pB-pA)
    return d

def PositionUnity2Ros(vector3):
    #vector3.z, -vector3.x, vector3.y);
    return [vector3[2], -vector3[0], vector3[1]]


def QuaternionUnity2Ros(quaternion):
    #return new Quaternion(-quaternion.z, quaternion.x, -quaternion.y, quaternion.w);
    return [ -quaternion[2], quaternion[0], -quaternion[1], quaternion[3]]

class HeadTracking:
    def __init__(self):    
            rospy.init_node('head_pose')

            self.scene_transform_topic = rospy.get_param("transform_topic", "/scene/transform")

            self.front_slot_publisher = rospy.Publisher("/front_slot", PoseStamped, queue_size=10)
            self.back_slot_publisher = rospy.Publisher("/back_slot", PoseStamped, queue_size=10)
            
            self.sub = rospy.Subscriber(self.scene_transform_topic, String, self.transform_cb)
            rospy.spin()

    def transform_cb(self, str_msg):
        np.set_printoptions(precision=3)
        data = json.loads(str_msg.data)

        transform_to_world = dict()
        for transform in data:
            name = transform["name"]

            p_x = transform['position']['z']
            p_y = -transform['position']['x']
            p_z = transform['position']['y']

            o_x = -transform['rotation']['z']
            o_y = transform['rotation']['x']
            o_z = -transform['rotation']['y']
            o_w = transform['rotation']['w']

            m ={}
            m["p"] = {}
            m["p"]["x"] = p_x
            m["p"]["y"] = p_y
            m["p"]["z"] = p_z
            m["q"] = {}
            m["q"]["x"] = o_x
            m["q"]["y"] = o_y
            m["q"]["z"] = o_z
            m["q"]["w"] = o_w

            transform_to_world[name]=m

        front_slot_pose = PoseStamped()
        back_slot_pose = PoseStamped()

        front_slot_pose.header.frame_id="dual_arm"
        back_slot_pose.header.frame_id="dual_arm"

        front_slot_pose.pose.position.x = transform_to_world[FRONT_SLOT_NAME]["p"]["x"]
        front_slot_pose.pose.position.y = transform_to_world[FRONT_SLOT_NAME]["p"]["y"]
        front_slot_pose.pose.position.z = transform_to_world[FRONT_SLOT_NAME]["p"]["z"]
        front_slot_pose.pose.orientation.x = transform_to_world[FRONT_SLOT_NAME]["q"]["x"]
        front_slot_pose.pose.orientation.y = transform_to_world[FRONT_SLOT_NAME]["q"]["y"]
        front_slot_pose.pose.orientation.z = transform_to_world[FRONT_SLOT_NAME]["q"]["z"]
        front_slot_pose.pose.orientation.w = transform_to_world[FRONT_SLOT_NAME]["q"]["w"]

        back_slot_pose.pose.position.x = transform_to_world[BACK_SLOT_NAME]["p"]["x"]
        back_slot_pose.pose.position.y = transform_to_world[BACK_SLOT_NAME]["p"]["y"]
        back_slot_pose.pose.position.z = transform_to_world[BACK_SLOT_NAME]["p"]["z"]
        back_slot_pose.pose.orientation.x = transform_to_world[BACK_SLOT_NAME]["q"]["x"]
        back_slot_pose.pose.orientation.y = transform_to_world[BACK_SLOT_NAME]["q"]["y"]
        back_slot_pose.pose.orientation.z = transform_to_world[BACK_SLOT_NAME]["q"]["z"]
        back_slot_pose.pose.orientation.w = transform_to_world[BACK_SLOT_NAME]["q"]["w"]

        self.front_slot_publisher.publish(front_slot_pose)
        self.back_slot_publisher.publish(back_slot_pose)
        
if __name__ == '__main__':
    track = HeadTracking()