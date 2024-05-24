#!/usr/bin/env python3

import rospy
import json
import tf
import tf2_ros
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
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
            '''
            self.br_left = tf2_ros.StaticTransformBroadcaster()
            self.br_right = tf2_ros.StaticTransformBroadcaster()
            '''
            self.transformer = tf.TransformerROS()
            self.t_old = rospy.Time.now()
            self.sub = rospy.Subscriber(self.scene_transform_topic, String, self.transform_cb)
            
            #msg = rospy.wait_for_message(self.scene_transform_topic, String)
            #self.transform_cb(msg)
            
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
        now = rospy.Time.now()

        left_transform = TransformStamped()  
        left_transform.header.stamp = now
        left_transform.header.frame_id = "dual_arm"
        left_transform.child_frame_id = "left_world"
        left_transform.transform.translation.x = l_p[0]
        left_transform.transform.translation.y = l_p[1]
        left_transform.transform.translation.z = l_p[2]
        left_transform.transform.rotation.x = l_q[0]
        left_transform.transform.rotation.y = l_q[1]
        left_transform.transform.rotation.z = l_q[2]
        left_transform.transform.rotation.w = l_q[3]

        right_transform = TransformStamped()
        right_transform.header.stamp = now
        right_transform.header.frame_id = "dual_arm"
        right_transform.child_frame_id = "right_world"
        right_transform.transform.translation.x = r_p[0]
        right_transform.transform.translation.y = r_p[1]
        right_transform.transform.translation.z = r_p[2]
        right_transform.transform.rotation.x = r_q[0]
        right_transform.transform.rotation.y = r_q[1]
        right_transform.transform.rotation.z = r_q[2]
        right_transform.transform.rotation.w = r_q[3]

        print(left_transform)
        print(right_transform)


        self.br_left.sendTransform(left_transform)
        self.br_right.sendTransform(right_transform)
        '''
        
        t = rospy.get_rostime()
        if t > (self.t_old + rospy.Duration(0.001)):
            #rospy.loginfo((t,l_p,l_q))
            #rospy.loginfo((t,r_p,r_q))
            self.br_left.sendTransform(translation=l_p, 
                                  rotation=l_q,
                                  time=t,
                                  child="left_base_link",
                                  parent="world")
            self.br_right.sendTransform(translation=r_p, 
                                  rotation=r_q,
                                  time=t,
                                  child="right_base_link",
                                  parent="world")
            self.t_old = t
        
        
if __name__ == '__main__':
    track = DualArmTransform()