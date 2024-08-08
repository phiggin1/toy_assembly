#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
import tf

class Arms:
    def __init__(self):
        rospy.init_node(f'bbox', anonymous=True)
        self.arm = rospy.get_param("~arm", "left")
        rospy.loginfo(f"arm: {self.arm}")

        self.base_frame = "world"
        self.padding = 0.075
        left_frames = [
            "left_base_link",
            "left_shoulder_link",
            "left_bicep_link",
            "left_forearm_link",
            "left_spherical_wrist_1_link",
            "left_spherical_wrist_2_link",
            "left_bracelet_link",
            "left_end_effector_link",
            "left_tool_frame"
        ]
        right_frames = [
            "right_base_link",
            "right_shoulder_link",
            "right_bicep_link",
            "right_forearm_link",
            "right_spherical_wrist_1_link",
            "right_spherical_wrist_2_link",
            "right_bracelet_link",
            "right_end_effector_link",
            "right_tool_frame"
        ]
        self.frames = {
            "left": left_frames,
            "right": right_frames
        }

        self.listener = tf.TransformListener()
        self.marker_pub = rospy.Publisher(f"{self.arm}_arm_bbox", Marker, queue_size=10)

        while not rospy.is_shutdown():
            '''
            left_marker = self.get_marker("left")
            left_marker.id = 0
            left_marker.color.r = 0.0

            right_marker = self.get_marker("right")
            left_marker.id = 1
            right_marker.color.g = 0.0

            marker_array = MarkerArray()
            marker_array.markers.append(left_marker)
            marker_array.markers.append(right_marker)
            self.marker_pub.publish(marker_array)
            '''
            marker = self.get_marker(self.arm)
            self.marker_pub.publish(marker)



    def get_marker(self, arm):
        points = []
        target = PointStamped()
        for frame in self.frames[arm]:
            self.listener.waitForTransform(frame, self.base_frame, rospy.Time(), rospy.Duration(4.0) )
            target.header.frame_id = frame
            target.header.stamp = rospy.Time()
            transformned_target = self.listener.transformPoint(self.base_frame, target)
            points.append((transformned_target.point,frame))

        marker = self.get_bbox(points)
        
        return marker
        
    def get_bbox(self, points):
        min_x =  99.9
        max_x = -99.9
        min_y =  99.9
        max_y = -99.9
        min_z =  99.9
        max_z = -99.9
        for p, frame in points:
            #print(f"{frame}: {p.x}, {p.y}, {p.z}")
            if p.x > max_x:
                max_x = p.x
            if p.x < min_x:
                min_x = p.x

            if p.y > max_y:
                max_y = p.y
            if p.y < min_y:
                min_y = p.y

            if p.z > max_z:
                max_z = p.z
            if p.z < min_z:
                min_z = p.z

        min_x -= self.padding
        max_x += self.padding
        min_y -= self.padding
        max_y += self.padding
        min_z -= self.padding
        max_z += self.padding

        print(f"{self.arm}: {min_x:,.4f}, {max_x:,.4f}, {min_y:,.4f}, {max_y:,.4f}, {min_z:,.4f}, {max_z:,.4f}")

        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.type = Marker.CUBE
        marker.pose.position.x = (max_x + min_x)/2.0
        marker.pose.position.y = (max_y + min_y)/2.0
        marker.pose.position.z = (max_z + min_z)/2.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = (max_x - min_x)
        marker.scale.y = (max_y - min_y)
        marker.scale.z = (max_z - min_z)

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.25

        return marker


if __name__ == '__main__':
    a = Arms()
