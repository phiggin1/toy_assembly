#!/usr/bin/env python3

import rospy
import json
import tf
import math
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose, PoseStamped

FRONT_NAME = "/horse_front_legs/FrontTarget"
BACK_NAME = "/horse_back_legs/BackTarget"

def PositionUnity2Ros(vector3):
    #vector3.z, -vector3.x, vector3.y);
    return [vector3[2], -vector3[0], vector3[1]]


def QuaternionUnity2Ros(quaternion):
    #return new Quaternion(-quaternion.z, quaternion.x, -quaternion.y, quaternion.w);
    return [ -quaternion[2], quaternion[0], -quaternion[1], quaternion[3]]

class HeadTracking:
    def __init__(self):    
            rospy.init_node('human_part_tacking')

            self.front_pose_publisher = rospy.Publisher("/front_part", PoseStamped, queue_size=10)
            self.back_pose_publisher = rospy.Publisher("/back_part", PoseStamped, queue_size=10)
            
            self.scene_transform_topic = rospy.get_param("transform_topic", "/scene/transform")
            self.sub = rospy.Subscriber(self.scene_transform_topic, String, self.transform_cb)
            rospy.spin()

    def transform_cb(self, str_msg):
        np.set_printoptions(precision=3)
        data = json.loads(str_msg.data)

        transform_to_world = dict()
        for transform in data:
            name = transform["name"]
            #print(name)

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

        front_pose = PoseStamped()
        back_pose = PoseStamped()

        front_pose.header.frame_id="dual_arm"
        back_pose.header.frame_id="dual_arm"

        front_pose.pose.position.x = transform_to_world[FRONT_NAME]["p"]["x"]
        front_pose.pose.position.y = transform_to_world[FRONT_NAME]["p"]["y"]
        front_pose.pose.position.z = transform_to_world[FRONT_NAME]["p"]["z"]
        front_pose.pose.orientation.x = transform_to_world[FRONT_NAME]["q"]["x"]
        front_pose.pose.orientation.y = transform_to_world[FRONT_NAME]["q"]["y"]
        front_pose.pose.orientation.z = transform_to_world[FRONT_NAME]["q"]["z"]
        front_pose.pose.orientation.w = transform_to_world[FRONT_NAME]["q"]["w"]

        back_pose.pose.position.x = transform_to_world[BACK_NAME]["p"]["x"]
        back_pose.pose.position.y = transform_to_world[BACK_NAME]["p"]["y"]
        back_pose.pose.position.z = transform_to_world[BACK_NAME]["p"]["z"]
        back_pose.pose.orientation.x = transform_to_world[BACK_NAME]["q"]["x"]
        back_pose.pose.orientation.y = transform_to_world[BACK_NAME]["q"]["y"]
        back_pose.pose.orientation.z = transform_to_world[BACK_NAME]["q"]["z"]
        back_pose.pose.orientation.w = transform_to_world[BACK_NAME]["q"]["w"]

        self.front_pose_publisher.publish(front_pose)
        self.back_pose_publisher.publish(back_pose)
        
if __name__ == '__main__':
    track = HeadTracking()