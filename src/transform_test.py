#!/usr/bin/env python3

import rospy
import json
import tf.transformations
import math
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point

#CAMERA = "/gen3_robotiq_2f_85_left/world/base_link/shoulder_link/bicep_link/forearm_link/spherical_wrist_1_link/spherical_wrist_2_link/bracelet_link/end_effector_link/camera_link/camera_standin"
CAMERA = "/Player/NoSteamVRFallbackObjects/FallbackObjects/camera"
OBJECT = "/horse_body"

def Unity2Ros(vector3):
    #return (vector3.z, -vector3.x, vector3.y);
    return (vector3[2], -vector3[0], vector3[1])

class TestTracker:
    def __init__(self):    
            rospy.init_node('transform')

            self.camera_name = rospy.get_param("camera_name", CAMERA)
            self.target_name = rospy.get_param("target_name", OBJECT)
            self.scene_transform_topic = rospy.get_param("transform_topic", "/scene/transform")
            self.target_point_topic = rospy.get_param("target_topic", "/pt")
            self.target_pose_topic = rospy.get_param("target_pose_topic", "/pose")

            self.sub = rospy.Subscriber(self.scene_transform_topic, String, self.transform_cb)
            self.pub = rospy.Publisher(self.target_point_topic, PointStamped, queue_size=10)
            rospy.spin()

    def transform_cb(self, str_msg):
        np.set_printoptions(precision=3)
        data = json.loads(str_msg.data)
        #print(data)
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
            roll, pitch, yaw = tf.transformations.euler_from_quaternion([o_x, o_y, o_z, o_w])

            m = tf.transformations.compose_matrix(None, None, (roll, pitch, yaw), (p_x,p_y,p_z), None)
            transform_to_world[name]=m

            rospy.loginfo(f"{name}")
            rospy.loginfo(f"{p_x},{p_y},{p_z}")
            #rospy.loginfo(roll,pitch,yaw)
            #rospy.loginfo(f"\n{m}")
            rospy.loginfo('-----')
            
            
        object_to_camera = np.matmul(transform_to_world[self.target_name], np.linalg.inv(transform_to_world[self.camera_name]))
        
        point = np.zeros( (4,1) )
        point[3,0]=1


        '''
        # mat44 is frame-to-frame transform as a 4x4
        mat44 = object_to_camera

        # pose44 is the given pose as a 4x4
        p = Point()
        q = Quat()
        position = tf.transformations.xyz_to_mat44(ps.pose.position)
        orientation = tf.transformations.xyzw_to_mat44(ps.pose.orientation)
        pose44 = numpy.dot(position, orientation)

        # txpose is the new pose in target_frame as a 4x4
        txpose = numpy.dot(mat44, pose44)

        # xyz and quat are txpose's position and orientation
        xyz = tuple(transformations.translation_from_matrix(txpose))[:3]
        quat = tuple(transformations.quaternion_from_matrix(txpose))
        '''

        object_position_in_camera = np.matmul(object_to_camera, point)

        x = object_position_in_camera[0][0]
        y = object_position_in_camera[1][0]
        z = object_position_in_camera[2][0]

        p = PointStamped()
        p.header.frame_id = "camera_link"
        p.point.x = x
        p.point.y = y
        p.point.z = z
        
        rospy.loginfo(p.point)
        rospy.loginfo("=====")
        
        self.pub.publish(p)
        
if __name__ == '__main__':
    track = TestTracker()