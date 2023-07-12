#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PointStamped
from toy_assembly.srv import Indicate

rospy.init_node('toy_assembly_main')

#get images of all the possible bodies
    #point cloud detect bodies (zone)
        #get locations of all possible bodies
    #focus on each body
    #SAM use center point of pointclout in image space as 'label'
        #get mask -> get rgb image -> get clip embdding
    #create list of dictionary {"image":"np array", "position":"PointStamped  of grasp target"}

#while not correct_selection
    #get the users description of the body they want
    #get users gaze during description (have long buffer of head pose or selected target with timestamps)
    #robot makes selection based on language + gaze
    #ask if correct
        #if not remove selection 
    #else
        #correct_selection

#manage the insertion process

serv = rospy.ServiceProxy('indicate', Indicate)
rospy.wait_for_service('indicate')


#points (x,y,z)
points = [
    [0.3, 0.0, 0.2 ],
    [0.3, -0.2, 0.2],
    [0.3, 0.2, 0.2]
]

for p in points:
    point = PointStamped()  #http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PointStamped.html
    point.point.x = p[0]
    point.point.y = p[1]
    point.point.z = p[2]
    
    print(point)
    resp = serv(point)
    print(resp)