#!/usr/bin/env python3

import rospy
import json
import tf.transformations
import math
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point

LEFT_CAMERA = "/gen3_robotiq_2f_85_left/world/base_link/shoulder_link/bicep_link/forearm_link/spherical_wrist_1_link/spherical_wrist_2_link/bracelet_link/end_effector_link/camera_link/camera_standin"
OBJECT = "/horse_body (2)"

def Unity2Ros(vector3):
    #return (vector3.z, -vector3.x, vector3.y);
    return (vector3[2], -vector3[0], vector3[1])

class TestTracker:
    def __init__(self):    
            rospy.init_node('transform')
            self.sub = rospy.Subscriber("/scene/tranform", String, self.transform_cb)
            self.pub = rospy.Publisher("/target_point", PointStamped, queue_size=10)
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

            '''
            print(name)
            print(p_x,p_y,p_z)
            print(roll,pitch,yaw)
            print(m)
            print('-----')
            '''
            
        object_to_camera = np.matmul(transform_to_world[OBJECT], np.linalg.inv(transform_to_world[LEFT_CAMERA]))
        
        point = np.zeros( (4,1) )
        point[3,0]=1

        object_position_in_camera = np.matmul(object_to_camera, point)

        x = object_position_in_camera[0][0]
        y = object_position_in_camera[1][0]
        z = object_position_in_camera[2][0]

        p = PointStamped()
        p.header.frame_id = "camera_link"
        p.point.x = x
        p.point.y = -y
        p.point.z = z
        '''
        print(p.point)
        print("=====")
        '''
        self.pub.publish(p)
        
if __name__ == '__main__':
    track = TestTracker()