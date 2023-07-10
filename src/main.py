#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PointStamped


rospy.init_node('toy_assembly_main')

#get images of all the possible bodies (SAM if positions known ahead of time can project points into image and get segmented images that way) 
#get locations of all possible bodies -> can this be preknown?

#dictionary of "id":{"image":"np array", "position":"PointStamped  of grasp target"}

#get the users description of the body they want
#get users gaze during description

#while not correct_selection
    #robot makes selection
        #language + gaze
    #ask if correct
        #get audio (yes/no)
        #get sentiment
    #if not remove selection 


serv = rospy.ServiceProxy('indicate', Indicate)


#points (x,y,z)
points = [
    [0.3, 0.0, 0.2 ],
    [0.3, -0.2, 0.2],
    [0.3, 0.2, 0.2]
]

for p in points:
    print(p)
    point = PointStamped()

    resp = serv(point)


