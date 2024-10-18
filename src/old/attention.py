#!/usr/bin/env python3

import numpy as np
import rospy
import math
from std_msgs.msg import Float32MultiArray
from toy_assembly.msg import Transcription
from toy_assembly.msg import Intrest

class Attention:
    def __init__(self):    
        rospy.init_node('attention')
        #sub gaze
        #sub bayesgaze
        #sub transcription

        #log  focus from gaze/bayes

        self.targets = []
         
        #use transcrip start and duration to slice focus
        #self.bayes_gaze_sub = rospy.Subscriber("/intrest", Float32MultiArray, self.bayes_gaze_cb)
        
        #self.gaze_sub = rospy.Subscriber("/distances", Float32MultiArray, self.gaze_cb)
        self.gaze_sub = rospy.Subscriber("/distances", Intrest, self.gaze_cb)
        self.transript_sub = rospy.Subscriber("/transcript", Transcription, self.transcript_cb)

        rospy.spin()



    def gaze_cb(self, distances):
        time = rospy.Time.now().to_sec()
        #data = [time, distances.data]
        data = [time, distances.intrest, distances.positions]
        self.targets.append(data)

        
    def bayes_gaze_cb(self, distances):
        time = rospy.Time.now().to_sec()
        #data = [time, distances.data]
        data = [time, distances.intrest, distances.positions]
        self.targets.append(data)

    def transcript_cb(self, transcription):
        end_time = transcription.audio_recieved.to_sec()
        print(end_time)
        duration = transcription.duration
        start_time = end_time-duration

        rospy.loginfo(f"duration:{duration}")
        rospy.loginfo(f"start_time:{start_time} to end_time:{end_time}")

        start_indx, end_indx = self.find_range(start_time,end_time)
        rospy.loginfo(f"{start_indx},{end_indx}")
        for i in range(start_indx,end_indx):
            print(self.targets[i])


    def find_range(self, start, end):
        start_indx = 0
        end_indx = len(self.targets)-1
        num_indexs = len(self.targets)
        init_time = self.targets[start_indx][0]
        final_time = self.targets[end_indx][0]
        rate = (final_time-init_time)/num_indexs

        '''
        print(f"init:  {init_time}")
        print(f"start: {start}")
        print(f"end:   {end}")
        print(f"final: {final_time}")
        print(f"range: {final_time-init_time}")
        print(f"num_indexs:{num_indexs}")
        print(f"rate:{rate}")
        '''

        start_guess = math.floor((start-init_time)/rate)
        #print(f"init guess:{start_guess}")
        found = False
        #break if timestampe is after start and prev is before
        while not found:
            if self.targets[start_guess][0] > start and self.targets[start_guess-1][0] < start:
                found = True
            #do better searxg scale add by diff
            elif self.targets[start_guess][0] > start:
                start_guess -= 1
            elif self.targets[start_guess][0] < start:
                start_guess += 1

        '''
        print(f"final guess:{start_guess}")
        print(self.targets[start_guess-1][0])
        print(self.targets[start_guess][0])
        print(self.targets[start_guess+1][0])
        print(start)
        '''

        end_guess = math.floor((end-init_time)/rate)
        #print(f"init end guess:{end_guess}")
        found = False
        #break if timestamp is after start and prev is before
        while not found:
            if self.targets[end_guess][0] < end and self.targets[end_guess+1][0] > end:
                found = True
            #do better searxg scale add by diff
            elif self.targets[end_guess][0] > end:
                end_guess -= 1
            elif self.targets[end_guess][0] < end:
                end_guess += 1

        '''
        print(f"final end guess:{end_guess}")
        print(self.targets[end_guess-1][0])
        print(self.targets[end_guess][0])
        print(self.targets[end_guess+1][0])
        print(end)
        '''

        return (start_guess, end_guess)


if __name__ == '__main__':
    track = Attention()
