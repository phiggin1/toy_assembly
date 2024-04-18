#!/usr/bin/env python3

import rospy
import tf
import math
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseStamped, Pose
from obj_segmentation.msg import SegmentedClustersArray
import sensor_msgs.point_cloud2 as pc2

def distance (a,b):
    dx = a[0]-b[0]
    dy = a[1]-b[1]
    dz = a[2]-b[2]

    return math.sqrt(dx*dx + dy*dy + dz*dz)

class RobotFilter:
    def __init__(self):
        rospy.init_node('robot_filter')

        self.object_clusters_topic = rospy.get_param("object_clusters_topic", "/unity/camera/left/depth/object_clusters")
        topic = self.object_clusters_topic.split("/")
        ns = "/".join(topic[0:-1])
        topic_name = topic[-1]
        filtered_topic_name = "/filtered_"+topic_name
        self.filtered_object_clusters_topic = ns+filtered_topic_name
        self.obj_cluster_pub = rospy.Publisher(self.filtered_object_clusters_topic, SegmentedClustersArray, queue_size=1)

        self.tf_listener = tf.TransformListener()

        self.threshold = 0.05

        self.robot_frames = [
            "right_base_link",
            "right_shoulder_link",
            "right_bicep_link",
            "right_spherical_wrist_1_link",
            "right_spherical_wrist_2_link",
            "right_bracelet_link",
            "right_end_effector_link",
            "right_right_inner_finger_pad",
            "right_left_inner_finger_pad"
        ]

        self.debug_pub = rospy.Publisher("/debug_array", PoseArray, queue_size=10)
        self.obj_cluster_sub = rospy.Subscriber(self.object_clusters_topic, SegmentedClustersArray, self.process_clusters)
        

        #rospy.spin()
    
    def process_clusters(self, clusters):
        cluster_frame = clusters.header.frame_id

        output_clusters = SegmentedClustersArray()
        output_clusters.header = clusters.header

        pose_array = PoseArray()
        pose_array.header = clusters.header
        robot_parts = []
        for frame in self.robot_frames:
            p1 = PoseStamped()
            p1.header.frame_id = frame
            p1.pose.orientation.w = 1.0
            now = rospy.Time.now()
            self.tf_listener.waitForTransform(cluster_frame, frame, now, rospy.Duration(4.0))
            p_in_base = self.tf_listener.transformPose(cluster_frame, p1)
            #pose_array.poses.append(p_in_base.pose)
            a = [
                p_in_base.pose.position.x,
                p_in_base.pose.position.y,
                p_in_base.pose.position.z
            ]
            robot_parts.append(a)

        #rospy.loginfo("a")
        for i, cluster in enumerate(clusters.clusters):
            #rospy.loginfo("b")
            min_dist = 99999.9
            frame = None

            for p in pc2.read_points(cluster, field_names = ("x", "y", "z"), skip_nans=True):
                for robot_part in robot_parts:
                    d = distance(robot_part,p)
                    if d < min_dist:
                        min_dist = d
                        frame = robot_part
                #break

            #rospy.loginfo(f"Cluster {i}: min_dist: {min_dist}, frame: {frame}")

            if min_dist > self.threshold:
                output_clusters.clusters.append(cluster)
                for p in pc2.read_points(cluster, field_names = ("x", "y", "z"), skip_nans=True):
                    pose = Pose()
                    pose.position.x = p[0]
                    pose.position.y = p[1]
                    pose.position.z = p[2]
                    pose.orientation.w = 1.0
                    pose_array.poses.append(pose)
                    break
            #else:
            #    print(f"Discarding Cluster {i}, frame: {frame}\n")
        
        self.debug_pub.publish(pose_array)
        #output_clusters.header.stamp = rospy.Time.now()
        self.obj_cluster_pub.publish(output_clusters)

if __name__ == '__main__':
    filter = RobotFilter()