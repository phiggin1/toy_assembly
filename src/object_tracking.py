#!/usr/bin/env python3

import rospy
import json
import tf
import math
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose

LEFT_CAMERA_FRAME="/gen3_robotiq_2f_85_left/world/base_link/shoulder_link/bicep_link/forearm_link/spherical_wrist_1_link/spherical_wrist_2_link/bracelet_link/end_effector_link/camera_link"
RIGHT_CAMERA_FRAME="/gen3_robotiq_2f_85_right/world/base_link/shoulder_link/bicep_link/forearm_link/spherical_wrist_1_link/spherical_wrist_2_link/bracelet_link/end_effector_link/camera_link"
OBJECTS_NAMES = ["/red_horse_front_legs",
                 "/yellow_horse_back_legs",
                 "/horse_body_blue",
                 "/horse_body_yellow",
                 "/horse_body_red"

]

def PositionUnity2Ros(vector3):
    #vector3.z, -vector3.x, vector3.y);
    return [vector3[2], -vector3[0], vector3[1]]


def QuaternionUnity2Ros(quaternion):
    #return new Quaternion(-quaternion.z, quaternion.x, -quaternion.y, quaternion.w);
    return [ -quaternion[2], quaternion[0], -quaternion[1], quaternion[3]]

class HeadTracking:
    def __init__(self):    
            rospy.init_node('object_poses')
            

            self.pose_array_publisher = rospy.Publisher("/toy_part_pose_array", PoseArray, queue_size=10)
            
            self.names_publisher = rospy.Publisher("/names", String, queue_size=10)

            self.scene_transform_topic = rospy.get_param("transform_topic", "/scene/transform")
            self.sub = rospy.Subscriber(self.scene_transform_topic, String, self.transform_cb)
            rospy.spin()

    def transform_cb(self, str_msg):
        np.set_printoptions(precision=3)
        data = json.loads(str_msg.data)

        transform_to_world = dict()
        for transform in data:
            name = transform["name"]
            '''
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

        names = []
        object_poses = PoseArray()
        object_poses.header.frame_id="dual_arm"
        for name in OBJECTS_NAMES:
            names.append(name)
            #rospy.loginfo(f"name:{name}, {transform_to_world[name]}")
            p = Pose()
            p.position.x = transform_to_world[name]["p"]["x"]
            p.position.y = transform_to_world[name]["p"]["y"]
            p.position.z = transform_to_world[name]["p"]["z"]
            p.orientation.x = transform_to_world[name]["q"]["x"]
            p.orientation.y = transform_to_world[name]["q"]["y"]
            p.orientation.z = transform_to_world[name]["q"]["z"]
            p.orientation.w = transform_to_world[name]["q"]["w"]
            object_poses.poses.append(p)

        ros_string = String()
        ros_string.data = ','.join(names)
        self.names_publisher.publish(ros_string)
        self.pose_array_publisher.publish(object_poses)
        
if __name__ == '__main__':
    track = HeadTracking()