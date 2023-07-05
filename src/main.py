#!/usr/bin/env python3

import sys
import rospy
from geometry_msgs.msg import PointStamped


rospy.init_node('toy_assembly_main')


points = [
    [0.3, 0.0, 0.2 ],
    [0.3, -0.2, 0.2],
    [0.3, 0.2, 0.2]
]

for p in points:
    print(p)