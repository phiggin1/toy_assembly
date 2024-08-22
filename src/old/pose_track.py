#!/usr/bin/env python3

import rospy
import numpy as np
from obj_segmentation.msg import SegmentedClustersArray
from obj_segmentation.msg import Object
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseArray, Pose, Quaternion


class PoseArrayTrack:
    def __init__(self):
        rospy.init_node('cluster_view', anonymous=True)
        self.pose_array_pub = rospy.Publisher("/object_poses", PoseArray, queue_size=10)
        self.obj_cluster_sub = rospy.Subscriber("/object_clusters", SegmentedClustersArray, self.process_clusters)
        rospy.spin()
    
    def process_clusters(self, req):
        #iterate through all the objects
        obj_position = []
        for i, pc in enumerate(req.clusters):
            print("obj %d" % i)
            min_x = 1000.0
            min_y = 1000.0
            min_z = 1000.0
            max_x = -1000.0
            max_y = -1000.0
            max_z = -1000.0

            #for each object get a bounding box
            for p in pc2.read_points(pc):
                if p[0] > max_x:
                    max_x = p[0]
                if p[0] < min_x:
                    min_x = p[0]

                if p[1] > max_y:
                    max_y = p[1]
                if p[1] < min_y:
                    min_y = p[1]

                if p[2] > max_z:
                    max_z = p[2]
                if p[2] < min_z:
                    min_z = p[2]

            center = [(min_x + max_x)/2, (min_y + max_y)/2, (min_z + max_z)/2]
            obj_position.append(center)

            q = Quaternion()
            q.w = 1.0

            pose_array = PoseArray()
            pose_array.header.frame_id = req.header.frame_id
            for o in obj_position:
                 #compare distance to part of robot
                 #if close skip
                 #else add to array

                 p = Pose()
                 p.position.x = o[0]
                 p.position.x = o[1]
                 p.position.x = o[2]
                 pose_array.poses.append(p)

            self.pose_array_pub.publish(pose_array)

	
if __name__ == '__main__':
    a = PoseArrayTrack()

