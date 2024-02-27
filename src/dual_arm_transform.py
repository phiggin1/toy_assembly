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
            self.t_old = rospy.Time.now()
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

            #converting from unity to ros
            m ={}
            m["p"] = {}
            m["p"]["x"] = p_z
            m["p"]["y"] = -p_x
            m["p"]["z"] = p_y
            m["q"] = {}
            m["q"]["x"] = -o_z
            m["q"]["y"] = o_x
            m["q"]["z"] = -o_y
            m["q"]["w"] = o_w


            #roll, pitch, yaw = tf.transformations.euler_from_quaternion([o_x, o_y, o_z, o_w])
            #m = tf.transformations.compose_matrix(None, None, (roll, pitch, yaw), (p_x,p_y,p_z), None)
            
            transform_to_world[name]=m

        #rospy.loginfo(f"{self.left_arm_frame}:\n{transform_to_world[self.left_arm_frame]}")
        left_p = (transform_to_world[self.left_arm_frame]["p"]["x"], 
                  transform_to_world[self.left_arm_frame]["p"]["y"], 
                  transform_to_world[self.left_arm_frame]["p"]["z"])
        left_q = (transform_to_world[self.left_arm_frame]["q"]["x"], 
                  transform_to_world[self.left_arm_frame]["q"]["y"], 
                  transform_to_world[self.left_arm_frame]["q"]["z"], 
                  transform_to_world[self.left_arm_frame]["q"]["w"])
        
        #rospy.loginfo(f"{self.right_arm_frame}:\n{transform_to_world[self.right_arm_frame]}")
        right_p = (transform_to_world[self.right_arm_frame]["p"]["x"], 
                   transform_to_world[self.right_arm_frame]["p"]["y"], 
                   transform_to_world[self.right_arm_frame]["p"]["z"])
        right_q = (transform_to_world[self.right_arm_frame]["q"]["x"], 
                   transform_to_world[self.right_arm_frame]["q"]["y"], 
                   transform_to_world[self.right_arm_frame]["q"]["z"], 
                   transform_to_world[self.right_arm_frame]["q"]["w"])

        p = [left_p[0]-right_p[0],
             left_p[1]-right_p[1],
             left_p[2]-right_p[2]
        ]
        q = [0.0,0.0,0.0,1.0]
        t = rospy.Time.now()
        #print(t)
        #print(self.t_old)
        #print('----')
        if t > (self.t_old + rospy.Duration(0.1)):
            rospy.loginfo((t,p,q))
            self.br.sendTransform(p, q,
                                  rospy.Time.now(),
                                  "left_base_link",
                                  "right_base_link")
            self.t_old = t

        
if __name__ == '__main__':
    track = DualArmTransform()