#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3



class Vel:
    def __init__(self):
        self.arm = "right"
        self.other_arm = "left"
        rospy.init_node(f'vel_scal_{self.arm}', anonymous=True)

        self.min_right = Vector3()
        self.max_right = Vector3()

        self.min_left = Vector3()
        self.max_left = Vector3()

        self.right_marker_sub = rospy.Subscriber(f"{self.arm}_arm_bbox", Marker, self.right_cb)
        self.left_marker_sub = rospy.Subscriber(f"{self.other_arm}_arm_bbox", Marker, self.left_cb)
        
        
        self.have_left = False
        self.have_right = False
        rate = rospy.Rate(10)
        while not self.have_left and not self.have_right:
            rate.sleep()

        while not rospy.is_shutdown():
            self.check_collision()
            rate.sleep()

        #rospy.spin()

    def check_collision(self):
        print(f"left  : {self.min_left.x:,.4f}, {self.max_left.x:,.4f}, {self.min_left.y:,.4f}, {self.max_left.y:,.4f}, {self.min_left.z:,.4f}, {self.max_left.z:,.4f}")
        print(f"right : {self.min_right.x:,.4f}, {self.max_right.x:,.4f}, {self.min_right.y:,.4f}, {self.max_right.y:,.4f}, {self.min_right.z:,.4f}, {self.max_right.z:,.4f}")
        if self.min_left.y < self.max_right.y:
            rospy.loginfo(f"possible collsion; {self.min_left.y - self.max_right.y}")
        else:
            rospy.loginfo(self.min_left.y - self.max_right.y)

    def right_cb(self, marker):
        x = marker.pose.position.x# = (max_x + min_x)/2.0
        y = marker.pose.position.y# = (max_y + min_y)/2.0
        z = marker.pose.position.z# = (max_z + min_z)/2.0
        depth  = marker.scale.x# = (max_x - min_x)
        width  = marker.scale.y# = (max_y - min_y)
        height = marker.scale.z# = (max_z - min_z)

        min_x = (2*x - depth)/2.0
        max_x = min_x + depth

        min_y = (2*y - width)/2.0
        max_y = min_y + width

        min_z = (2*z - height)/2.0
        max_z = min_z + height

        #print(f"right: {self.min_right.x:,.4f}, {self.max_right.x:,.4f}, {self.min_right.y:,.4f}, {self.max_right.y:,.4f}, {self.min_right.z:,.4f}, {self.max_right.z:,.4f}")
        self.min_right.x = min_x
        self.min_right.y = min_y
        self.min_right.z = min_z
        self.max_right.x = max_x
        self.max_right.y = max_y
        self.max_right.z = max_z

        self.have_left = True

    def left_cb(self, marker):
        x = marker.pose.position.x# = (max_x + min_x)/2.0
        y = marker.pose.position.y# = (max_y + min_y)/2.0
        z = marker.pose.position.z# = (max_z + min_z)/2.0
        depth  = marker.scale.x# = (max_x - min_x)
        width  = marker.scale.y# = (max_y - min_y)
        height = marker.scale.z# = (max_z - min_z)

        min_x = (2*x - depth)/2.0
        max_x = min_x + depth

        min_y = (2*y - width)/2.0
        max_y = min_y + width

        min_z = (2*z - height)/2.0
        max_z = min_z + height

        self.min_left.x = min_x
        self.min_left.y = min_y
        self.min_left.z = min_z
        self.max_left.x = max_x
        self.max_left.y = max_y
        self.max_left.z = max_z

        self.have_right = True

if __name__ == '__main__':
    a = Vel()
