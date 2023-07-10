#!/usr/bin/env python3

import rospy
import json
import numpy as np
import io
from std_msgs.msg import String
from scipy.io.wavfile import write
from toy_assembly.srv import Transcription

def is_silent(snd_data, threshold):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < threshold


class AudioSpeechToText:
    def __init__(self):
        rospy.init_node('AudioSpeechToText', anonymous=True)

        self.silent_wait = 10
        self.max_duration = 5
        self.threshold = 0.1
        self.rate = 16000

        self.num_silent = 0
        self.snd_started = False
        self.audio_clip = []

        #Virtual robot in RIVR or phyical robot
        #   determines which topics to subscribe too
        self.is_rivr = rospy.get_param("~rivr", True)

        #print('waiting for service')
        #rospy.wait_for_service('get_transciption')
        #print('got service')
        #self.serv = rospy.ServiceProxy('get_transciption', Transcription)

        if self.is_rivr:
            rospy.loginfo("virtual robot")
            self.audio_subscriber = rospy.Subscriber("/test", String, self.virtual_audio_cb)
        else:
            rospy.loginfo("physical robot")
            self.audio_subscriber = rospy.Subscriber("/test", String, self.physical_audio_cb)

        rospy.spin()
    
    def virtual_audio_cb(self, msg):
        data = json.loads(msg.data)
        self.process_audio(data)
    
    def physical_audio_cb(self, msg):
        #mp3 bytesdata data to normalized np float array



        print(msg)

    def process_audio(self, data):
        silent = is_silent(data, self.threshold)

        if not silent:
            rospy.loginfo("there be sound")

        self.audio_clip.extend(data)

        if silent and self.snd_started:
            self.num_silent += 1
        elif not silent and not self.snd_started:
            self.snd_started = True
            self.num_silent = 0
            self.audio_clip = []

        if self.snd_started and self.num_silent > self.silent_wait:
            print("got audio clip")
            self.get_transcription(self.audio_clip)
        
        '''
        add in some code for potential recording visualization aides
        if snd_started and not silent -> ?green activly lisenting
        elif snd_started and num_silent > 0 -> ?yellow thinks speaker is finished but unsure
        else -> ?red waiting for someone to talk
        '''

        #if the audio clip is over max duration get the transcription
        #clip is len(audio_clip)/rate seconds long
        if len(self.audio_clip)>(self.rate*self.max_duration) and self.snd_started:
            print("max clip length")
            #self.get_transcription(self.audio_clip)

    def get_transcription(self, audio):
            #get audio into correct format
            data = np.asarray(audio)
            bytes_wav = bytes()
            byte_io = io.BytesIO(bytes_wav)
            write(byte_io, self.rate, data)
            wav_data = byte_io.read()

            #call transcription service here
            print(data[0])




            self.snd_started = False
            self.num_silent = 0
            self.audio_clip = []


if __name__ == '__main__':
    get_target = AudioSpeechToText()
