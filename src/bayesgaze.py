#!/usr/bin/env python3
import numpy as np
import rospy
import math
from std_msgs.msg import Float32MultiArray

LEFT_CAMERA_FRAME="/gen3_robotiq_2f_85_left/world/base_link/shoulder_link/bicep_link/forearm_link/spherical_wrist_1_link/spherical_wrist_2_link/bracelet_link/end_effector_link/camera_link"
RIGHT_CAMERA_FRAME="/gen3_robotiq_2f_85_right/world/base_link/shoulder_link/bicep_link/forearm_link/spherical_wrist_1_link/spherical_wrist_2_link/bracelet_link/end_effector_link/camera_link"
OBJECTS_NAMES = [LEFT_CAMERA_FRAME,
                 RIGHT_CAMERA_FRAME, 
                 "/horse_body_red", 
                 "/horse_body_yellow", 
                "/horse_body_blue"
]

#eq 4) P(t_i)= (k+c_i) / k·N+sum(fromj=1 to N)(c_j)
def eq4(k, c, N, t):
    return (k+c[t])/(k*N + np.sum(c) )

#eq 5) P(s_i|t) = [1/sqrt(2πσ)]*exp[- (||s_i − c_t||^2) / (2σ^2) ]
def eq5(c, sigma):
    a = 1/(math.sqrt(2*math.pi*sigma*sigma))
    #c = cos_distance(head_gaze, obj_pos-head_pos)
    d = (2*sigma*sigma)
    b = math.exp(-(c*c)/d)

    return a*b

class BayesGaze:
    def __init__(self, k, theta, sigma, objects):
        rospy.init_node('bayesgaze')

        self.objects = objects

        self.k = k
        self.theta = theta
        self.sigma = sigma

        self.N = len(objects)
        self.c = np.zeros(self.N, dtype=np.int32)
        self.p = np.ones(self.N)/np.linalg.norm(np.ones(self.N))

        self.positions_list = []
        self.timestamps = []
        self.prev_intrest = np.zeros(self.N)
        self.i = 0

        self.sub  = rospy.Subscriber("/distances", Float32MultiArray, self.callback)
        rospy.spin()

    def callback(self, msg):
        np.set_printoptions(precision=5)
        stamp = rospy.Time.now().to_sec()
        object_distances = msg.data

        intrest = np.zeros(self.N)
        self.timestamps.append(stamp)

        if self.i>0:
            dt = self.timestamps[self.i]-self.timestamps[self.i-1]
        else:
            dt = 0.0

        p_of_si_given = np.zeros(self.N)
        p_of_si_given_sum = 0.0
        for t_j, obj in enumerate(self.objects):
            obj_distance = object_distances[t_j]
            p_of_si_given[t_j] = eq5(obj_distance, self.sigma)
            p_of_si_given_sum += p_of_si_given[t_j]*self.p[t_j]

        p_of_t_given_si = np.zeros(self.N)
        for t_j, obj in enumerate(self.objects):
            p_of_t_given_si[t_j] = (p_of_si_given[t_j]*self.p[t_j])/(p_of_si_given_sum)
            if self.i > 1:
                intrest[t_j] = self.prev_intrest[t_j] + dt*p_of_t_given_si[t_j]
            else:
                intrest[t_j] = dt*p_of_t_given_si[t_j]

        rospy.loginfo(f"c:{self.c}")
        rospy.loginfo(f"old:{self.prev_intrest}")
        rospy.loginfo(f"new:{intrest}")

        self.prev_intrest += intrest

        #if overthreshold mark object as selected
        target = np.argmax(self.prev_intrest)
        rospy.loginfo(f"update:{self.prev_intrest}")

        if (self.prev_intrest[target]) > self.theta:
            rospy.loginfo(f"target:{self.objects[target]}")
            self.prev_intrest = np.zeros(self.N)
            self.c[target] += 1
            for t_ji, obj in enumerate(self.objects):
                self.p[t_ji] = eq4(self.k, self.c, self.N, t_ji)
                #self.p=np.ones(self.N)/np.linalg.norm(np.ones(self.N))      

        self.i+=1

if __name__ == '__main__':
    sigma = 0.04
    theta =0.8
    k=1.0

    track = BayesGaze(k, theta, sigma, OBJECTS_NAMES)