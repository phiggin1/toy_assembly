#!/usr/bin/env python3

import rospy
import json
import math
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

HEAD_FRAME="/Player/NoSteamVRFallbackObjects/FallbackObjects/FollowHead"

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

            self.head_pose_publisher = rospy.Publisher("/head_pose", PoseStamped, queue_size=10)
            
            self.sub = rospy.Subscriber(self.scene_transform_topic, String, self.transform_cb)
            rospy.spin()

    def transform_cb(self, str_msg):
        np.set_printoptions(precision=3)
        data = json.loads(str_msg.data)

        transform_to_world = dict()
        for transform in data:
            name = transform["name"]

            p_x = transform['position']['x']
            p_y = transform['position']['y']
            p_z = transform['position']['z']

            o_x = transform['rotation']['x']
            o_y = transform['rotation']['y']
            o_z = transform['rotation']['z']
            o_w = transform['rotation']['w']
            
            '''
            p_x = transform['position']['z']
            p_y = -transform['position']['x']
            p_z = transform['position']['y']

            o_x = -transform['rotation']['z']
            o_y = transform['rotation']['x']
            o_z = -transform['rotation']['y']
            o_w = transform['rotation']['w']
            '''

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

        rospy.loginfo(transform_to_world[HEAD_FRAME])
        head_pose = PoseStamped()
        head_pose.header.frame_id="dual_arm"
        head_pose.pose.position.x = transform_to_world[HEAD_FRAME]["p"]["x"]
        head_pose.pose.position.y = transform_to_world[HEAD_FRAME]["p"]["y"]
        head_pose.pose.position.z = transform_to_world[HEAD_FRAME]["p"]["z"]
        head_pose.pose.orientation.x = transform_to_world[HEAD_FRAME]["q"]["x"]
        head_pose.pose.orientation.y = transform_to_world[HEAD_FRAME]["q"]["y"]
        head_pose.pose.orientation.z = transform_to_world[HEAD_FRAME]["q"]["z"]
        head_pose.pose.orientation.w = transform_to_world[HEAD_FRAME]["q"]["w"]
        self.head_pose_publisher.publish(head_pose)
        
if __name__ == '__main__':
    track = HeadTracking()