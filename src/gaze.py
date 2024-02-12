#!/usr/bin/env python3

import rospy
import message_filters
import json
import tf
import math
import numpy as np
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from scipy.spatial.distance import cosine as cos_distance
from scipy.spatial.transform import Rotation as R
from tf.transformations import euler_from_quaternion, quaternion_from_euler

LEFT_CAMERA_FRAME="/gen3_robotiq_2f_85_left/world/base_link/shoulder_link/bicep_link/forearm_link/spherical_wrist_1_link/spherical_wrist_2_link/bracelet_link/end_effector_link/camera_link"
RIGHT_CAMERA_FRAME="/gen3_robotiq_2f_85_right/world/base_link/shoulder_link/bicep_link/forearm_link/spherical_wrist_1_link/spherical_wrist_2_link/bracelet_link/end_effector_link/camera_link"
OBJECTS_NAMES = [LEFT_CAMERA_FRAME,
                 RIGHT_CAMERA_FRAME, 
                 "/horse_body_red", 
                 "/horse_body_yellow", 
                "/horse_body_blue"
]
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
            rospy.init_node('gaze')
            

            self.head_pose_topic = rospy.get_param("head_pose_topic", "/head_pose")
            self.object_pose_topic = rospy.get_param("object_pose_topic", "/pose_array")


            self.head_point_pub = rospy.Publisher("/head_point", PointStamped, queue_size=10)
            self.gaze_point_pub = rospy.Publisher("/gaze_point", PointStamped, queue_size=10)

            self.head_pose_sub = message_filters.Subscriber(self.head_pose_topic, PoseStamped)
            self.object_pose_sub = message_filters.Subscriber(self.object_pose_topic, PoseArray)
            self.ts = message_filters.TimeSynchronizer([self.head_pose_sub, self.object_pose_sub], 10)
            self.ts.registerCallback(self.callback)

            rospy.spin()

    def callback(self, head_pose, object_poses):
        names = OBJECTS_NAMES
        np.set_printoptions(precision=3)

        head_pos = np.asarray( [head_pose.pose.position.x,
                                head_pose.pose.position.y,
                                head_pose.pose.position.z])
        
        gaze = np.asarray( [head_pose.pose.orientation.x,
                            head_pose.pose.orientation.y,
                            head_pose.pose.orientation.z,
                            head_pose.pose.orientation.w])
        #gaze = euler_from_quaternion(gaze[0], gaze[1], gaze[2], gaze[3])
        gaze = R.from_quat(gaze).apply([1.0, 0.0, 0.0])
        distances = []
        for i, obj_pose in enumerate(object_poses.poses):
            position = np.asarray([obj_pose.position.x,
                                   obj_pose.position.y,
                                   obj_pose.position.z])

            head_point = PointStamped()
            gaze_point = PointStamped()
            head_point.header=head_pose.header
            gaze_point.header=head_pose.header
            head_point.point.x=head_pos[0]
            head_point.point.y=head_pos[1]
            head_point.point.z=head_pos[2]
            gaze_point.point.x=head_pos[0]+gaze[0]
            gaze_point.point.y=head_pos[1]+gaze[1]
            gaze_point.point.z=head_pos[2]+gaze[2]
            head_point.header.stamp=rospy.Time.now()
            gaze_point.header.stamp=rospy.Time.now()
            self.head_point_pub.publish(head_point)
            self.gaze_point_pub.publish(gaze_point)

            d = cos_distance(gaze, position-head_pos)
            distances.append((names[i],d))
            #rospy.loginfo(f"name:{names[i]}, distance:{d}")

        distances = np.asarray(distances)
        sorted_indxs = np.argsort(np.asarray(distances)[:,1])
        rospy.loginfo(f"The tuple with minimum value at index 1{distances[sorted_indxs[0],:]}")
        
if __name__ == '__main__':
    track = HeadTracking()