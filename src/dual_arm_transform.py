#!/usr/bin/env python3

import rospy
import json
import tf
import math
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point

LEFT_ARM = "/gen3_robotiq_2f_85_left"
RIGHT_ARM = "/gen3_robotiq_2f_85_right"
DUAL_ARM_BASE_FRAME = "/dual_arm"

class DualArmTransform:
    def __init__(self):    
            rospy.init_node('transform')

            self.left_arm_frame = rospy.get_param("left_arm_frame", LEFT_ARM)
            self.right_arm_frame = rospy.get_param("right_arm_frame", RIGHT_ARM)
            self.scene_transform_topic = rospy.get_param("transform_topic", "/scene/transform")
            
            self.br = tf.TransformBroadcaster()

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

            #roll, pitch, yaw = tf.transformations.euler_from_quaternion([o_x, o_y, o_z, o_w])
            #m = tf.transformations.compose_matrix(None, None, (roll, pitch, yaw), (p_x,p_y,p_z), None)
            
            transform_to_world[name]=m

        rospy.loginfo(f"{self.left_arm_frame}:\n{transform_to_world[self.left_arm_frame]}")
        left_p = (transform_to_world[self.left_arm_frame]["p"]["x"], 
                  transform_to_world[self.left_arm_frame]["p"]["y"], 
                  transform_to_world[self.left_arm_frame]["p"]["z"])
        left_q = (transform_to_world[self.left_arm_frame]["q"]["x"], 
                  transform_to_world[self.left_arm_frame]["q"]["y"], 
                  transform_to_world[self.left_arm_frame]["q"]["z"], 
                  transform_to_world[self.left_arm_frame]["q"]["w"])
        self.br.sendTransform(left_p, left_q,
                                  rospy.Time.now(),
                                  self.left_arm_frame,
                                  DUAL_ARM_BASE_FRAME)
        
        rospy.loginfo(f"{self.right_arm_frame}:\n{transform_to_world[self.right_arm_frame]}")
        right_p = (transform_to_world[self.right_arm_frame]["p"]["x"], 
                   transform_to_world[self.right_arm_frame]["p"]["y"], 
                   transform_to_world[self.right_arm_frame]["p"]["z"])
        right_q = (transform_to_world[self.right_arm_frame]["q"]["x"], 
                   transform_to_world[self.right_arm_frame]["q"]["y"], 
                   transform_to_world[self.right_arm_frame]["q"]["z"], 
                   transform_to_world[self.right_arm_frame]["q"]["w"])
        self.br.sendTransform(right_p, right_q,
                                  rospy.Time.now(),
                                  self.right_arm_frame,
                                  DUAL_ARM_BASE_FRAME)
        
if __name__ == '__main__':
    track = DualArmTransform()