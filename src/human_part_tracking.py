#!/usr/bin/env python3

import rospy
import json
import math
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseArray


def PositionUnity2Ros(vector3):
    #vector3.z, -vector3.x, vector3.y);
    return ( vector3[2], -vector3[0], vector3[1] )


def QuaternionUnity2Ros(quaternion):
    #return new Quaternion(-quaternion.z, quaternion.x, -quaternion.y, quaternion.w);
    return ( -quaternion[2], quaternion[0], -quaternion[1], quaternion[3] )

class HeadTracking:
    def __init__(self):    
            rospy.init_node('human_part_slot_pose')

            self.scene_transform_topic = rospy.get_param("transform_topic", "/scene/transform")

            self.slot_array_pub = rospy.Publisher("human_slot_array", PoseArray,  queue_size=10)
            self.obj_name= None

            
            rospy.loginfo("waiting for human_text_topic")
            self.text_topic = rospy.get_param("/human_text_topic", "human_text_topic") 
            self.sub = rospy.Subscriber(self.text_topic, String, self.object_cb)
            #self.obj_name = rospy.wait_for_message(self.text_topic, String)
            #self.obj_name = "/"+self.obj_name.data
            #rospy.loginfo("human: "+self.obj_name)

            self.sub = rospy.Subscriber(self.scene_transform_topic, String, self.transform_cb)
            rospy.spin()

    def object_cb(self, text):
        self.obj_name = "/"+text.data
        rospy.loginfo("human: "+self.obj_name)

    def transform_cb(self, str_msg):
        if self.obj_name is None:
            return
        np.set_printoptions(precision=3)
        data = json.loads(str_msg.data)

        transform_to_world = dict()
        for transform in data:
            name = transform["name"]
            if ((self.obj_name in name) and ("Slot" in name)):
                #print(self.obj_name, name)
                #going from unity (y up0 to ros (z up)
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

        #print("---------")


        pose_array = PoseArray()
        pose_array.header.frame_id = "dual_arm"
        for slot in transform_to_world:
            pose = Pose()
            pose.position.x = transform_to_world[slot]["p"]["x"]
            pose.position.y = transform_to_world[slot]["p"]["y"]
            pose.position.z = transform_to_world[slot]["p"]["z"]
            pose.orientation.x = transform_to_world[slot]["q"]["x"]
            pose.orientation.y = transform_to_world[slot]["q"]["y"]
            pose.orientation.z = transform_to_world[slot]["q"]["z"]
            pose.orientation.w = transform_to_world[slot]["q"]["w"]
            pose_array.poses.append(pose)

        #rospy.loginfo(f"human part: {pose_array.poses[0]}")
        #rospy.loginfo(f"human len: {len(pose_array.poses)}")
        self.slot_array_pub.publish(pose_array)



if __name__ == '__main__':
    track = HeadTracking()