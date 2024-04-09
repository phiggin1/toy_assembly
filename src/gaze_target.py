#!/usr/bin/env python3

import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from toy_assembly.msg import Intrest
from toy_assembly.msg import Transcription
from multiprocessing import Lock
from scipy.special import softmax

OBJECTS_NAMES = ["/horse_body_red", 
                 "/horse_body_yellow", 
                 "/horse_body_blue",
                 "/red_horse_front_legs",
                 "/yellow_horse_front_legs"
]

class GetGazeTarget:
    def __init__(self):    
        rospy.init_node('GetGazeTarget')
        self.mutex = Lock()
        self.max_length = 30*360 #~30fps times 6 minues
        self.distances = []
        self.dist_sub = rospy.Subscriber("/distances", Intrest, self.dist_cb)
        self.transript_sub = rospy.Subscriber("/transcript", Transcription, self.transcript_cb)

        self.gaze_target_pub = rospy.Publisher("/gaze_target", Float32MultiArray, queue_size=10)

        '''
        self.transript_pub = rospy.Publisher("/transcript", Transcription, queue_size=10)
        t = Transcription()
        t.duration = 5.0
        t.audio_recieved = rospy.Time.now()+rospy.Duration(8)
        t.transcription = "testing"
        rospy.loginfo(t)
        rospy.sleep(10)
        rospy.loginfo(t)
        self.transript_pub.publish(t)
        '''


        rospy.spin()

    def dist_cb(self, distances):
        #rospy.loginfo(len(self.distances))
        with self.mutex:
            now = rospy.Time.now()
            self.distances.append((now, distances.intrest))
            if len(self.distances) > self.max_length:
                self.distances.pop(0)
                rospy.loginfo("len(distances) > max_length")
         
    def transcript_cb(self, transcript):
        text = transcript.transcription
        end_time = transcript.audio_recieved.to_sec()
        start_time = end_time-transcript.duration
        print(text, start_time, end_time)
        '''
        t = Transcription()
        t.audio_recieved = now
        t.duration = duration
        t.transcription = transcript.transcription
        '''
        with self.mutex:
            print(len(self.distances))
            start_indx = -1 
            for i in range(len(self.distances)):
                #print(self.distances[i][0].to_sec())
                if (self.distances[i][0].to_sec()>start_time):
                    start_indx=i
                    break
            print(start_indx)
            if start_indx == -1:
                rospy.loginfo("no data")
                return

            #running_distances = np.zeros(len(self.distances[0][1]))
            running_targets = np.zeros(len(self.distances[0][1]))
            count = 0
            for i in range(start_indx, len(self.distances)):

                if self.distances[i][0].to_sec() > end_time:
                    break
                #for j in range(len(running_distances)):
                #    running_distances[j]+=self.distances[i][1][j]

                target = np.argmin(self.distances[i][1])
                running_targets[target] += 1
                count+=1

            print(OBJECTS_NAMES)
            print(running_targets/count)
            float_array = Float32MultiArray()
            float_array.data = running_targets/count
            self.gaze_target_pub.publish(float_array)


            '''
            print(running_distances)
            print(softmax(running_distances))
            norm_running_distances = running_distances/np.max(running_distances)
            print(norm_running_distances)
            print(softmax(norm_running_distances))
            one_minus_norm_running_distances = 1 -norm_running_distances
            print(one_minus_norm_running_distances)
            print(softmax(one_minus_norm_running_distances))
            '''


            self.distances = []
            
            
             
if __name__ == '__main__':
    track = GetGazeTarget()