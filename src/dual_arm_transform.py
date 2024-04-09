#!/usr/bin/env python3

import rospy
import json
import tf
import math
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point

import tf 

LEFT_ARM = "/gen3_robotiq_2f_85_left"
RIGHT_ARM = "/gen3_robotiq_2f_85_right"
DUAL_ARM_BASE_FRAME = "/dual_arm"

class DualArmTransform:
    def __init__(self):    
            rospy.init_node('transform')

            self.left_arm_frame = rospy.get_param("left_arm_frame", LEFT_ARM)
            self.right_arm_frame = rospy.get_param("right_arm_frame", RIGHT_ARM)
            self.scene_transform_topic = rospy.get_param("transform_topic", "/scene/transform")
            
            self.br_left = tf.TransformBroadcaster()
            self.br_right = tf.TransformBroadcaster()
            self.transformer = tf.TransformerROS()
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
        left_p = [transform_to_world[self.left_arm_frame]["p"]["x"], 
                  transform_to_world[self.left_arm_frame]["p"]["y"], 
                  transform_to_world[self.left_arm_frame]["p"]["z"]]
        left_q = [transform_to_world[self.left_arm_frame]["q"]["x"], 
                  transform_to_world[self.left_arm_frame]["q"]["y"], 
                  transform_to_world[self.left_arm_frame]["q"]["z"], 
                  transform_to_world[self.left_arm_frame]["q"]["w"]]
        world_to_left = self.transformer.fromTranslationRotation(translation=left_p, rotation=left_q)
        
        #rospy.loginfo(f"{self.right_arm_frame}:\n{transform_to_world[self.right_arm_frame]}")
        right_p = [transform_to_world[self.right_arm_frame]["p"]["x"], 
                   transform_to_world[self.right_arm_frame]["p"]["y"], 
                   transform_to_world[self.right_arm_frame]["p"]["z"]]
        right_q = [transform_to_world[self.right_arm_frame]["q"]["x"], 
                   transform_to_world[self.right_arm_frame]["q"]["y"], 
                   transform_to_world[self.right_arm_frame]["q"]["z"], 
                   transform_to_world[self.right_arm_frame]["q"]["w"]]
        world_to_right = self.transformer.fromTranslationRotation(translation=right_p, rotation=right_q)


        l_p = tf.transformations.translation_from_matrix(world_to_left)
        l_q = tf.transformations.quaternion_from_matrix(world_to_left)

        r_p = tf.transformations.translation_from_matrix(world_to_right)
        r_q = tf.transformations.quaternion_from_matrix(world_to_right)

        '''
        left_to_world = tf.transformations.inverse_matrix(world_to_left)
        right_to_world = tf.transformations.inverse_matrix(world_to_right)

        l_p = tf.transformations.translation_from_matrix(left_to_world)
        l_q = tf.transformations.quaternion_from_matrix(left_to_world)

        r_p = tf.transformations.translation_from_matrix(right_to_world)
        r_q = tf.transformations.quaternion_from_matrix(right_to_world)
        '''

        t = rospy.get_rostime()
        if t > (self.t_old + rospy.Duration(0.05)):
            #rospy.loginfo((t,l_p,l_q))
            #rospy.loginfo((t,r_p,r_q))
            self.br_left.sendTransform(translation=l_p, 
                                  rotation=l_q,
                                  time=t,
                                  child="left_world",
                                  parent="dual_arm")
            self.br_right.sendTransform(translation=r_p, 
                                  rotation=r_q,
                                  time=t,
                                  child="right_world",
                                  parent="dual_arm")
            self.t_old = t

        '''
        #left_to_right = tf.transformations.concatenate_matrices(left_to_world, world_to_right)
        right_to_left = tf.transformations.concatenate_matrices(right_to_world, world_to_left)

        #print(right_to_left)

        p = tf.transformations.translation_from_matrix(right_to_left)
        q = tf.transformations.quaternion_from_matrix(right_to_left)

        t = rospy.Time.now()
        if t > (self.t_old + rospy.Duration(0.1)):
            rospy.loginfo((t,p,q))
            self.br.sendTransform(translation=p, 
                                  rotation=q,
                                  time=rospy.Time.now(),
                                  child="left_base_link",
                                  parent="right_base_link")
            self.t_old = t

        '''
if __name__ == '__main__':
    track = DualArmTransform()