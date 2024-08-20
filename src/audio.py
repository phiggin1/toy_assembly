#!/usr/bin/env python3

import rospy
import json
import numpy as np
import io
from std_msgs.msg import String
from toy_assembly.srv import Whisper, WhisperRequest
from toy_assembly.msg import Transcription
from rivr_ros.msg import RivrAudio

def is_silent(snd_data, threshold):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < threshold


class AudioSpeechToText:
    def __init__(self):
        rospy.init_node('AudioSpeechToText', anonymous=True)

        self.debug = rospy.get_param("~debug", False) 

        #Number of audio messages that are below threshold 
        #   to determine if person stopped talking
        self.silent_wait = rospy.get_param("~silent_wait", 3)

        #Maximum audio clip length (seconds)
        self.max_duration = rospy.get_param("~max_duration", 5)

        #Threshold to detect when there is sound 
        self.threshold = rospy.get_param("~threshold", 0.01)

        rospy.loginfo("debug: %r"%self.debug)
        rospy.loginfo("silent_wait: %i"%self.silent_wait )
        rospy.loginfo("max_duration: %i"%self.max_duration )
        rospy.loginfo("threshold: %f"%self.threshold )

        #counter for number of silent audio messages
        self.num_silent = 0
        self.snd_started = False
        self.audio_clip = []
        self.prev = []
        #self.whisper_serv = rospy.ServiceProxy('get_transciption', Whisper)

        self.utterance_publisher = rospy.Publisher("/utterance", RivrAudio, queue_size=10)
        self.transript_publisher = rospy.Publisher("/transcript", Transcription, queue_size=10)
        self.status_pub = rospy.Publisher("/status", String, queue_size=10)

        self.audio_subscriber = rospy.Subscriber("/audio", RivrAudio, self.audio_cb)
        print(self.audio_subscriber )
        rospy.spin()
    
    def audio_cb(self, msg):
        float_array = list(msg.data)
        self.sample_rate = msg.sample_rate
        #if self.debug: rospy.loginfo(f"msg recv, max volumn:{max(float_array)}, sample_rate:{self.sample_rate}")
        self.process_audio(float_array)
    
    def process_audio(self, data):
        silent = is_silent(data, self.threshold)
        if not silent:
            rospy.loginfo(f"there is sound, max volumn:{max(data)}")

        if not silent and not self.snd_started:
            self.snd_started = True
            self.num_silent = 0
            self.audio_clip = self.prev
            self.status_pub.publish("LISTENING")
        elif silent and self.snd_started:
            self.num_silent += 1
        elif not silent and self.snd_started:
            self.num_silent = 0           
            #if self.debug: rospy.loginfo(f"num_silent:{self.num_silent}")

        if self.snd_started:
            self.audio_clip.extend(data)

        if self.snd_started and self.num_silent > self.silent_wait:     #enough quite time that they stopped speaking
            if self.debug: rospy.loginfo(f"got audio clip, num_silent:{self.num_silent}")
            self.get_utterence()
        elif self.snd_started and len(self.audio_clip)>(self.sample_rate*self.max_duration) :    #hit the maxiumum clip duration
            if self.debug: rospy.loginfo("max clip length")
            self.get_utterence()

        self.prev = data

    def get_utterence(self):
        self.status_pub.publish("THINKING")
        utterance = RivrAudio()
        utterance.sample_rate = self.sample_rate
        utterance.data = self.audio_clip
        self.utterance_publisher.publish(utterance)

        self.snd_started = False
        self.num_silent = 0
        self.audio_clip = []


if __name__ == '__main__':
    get_target = AudioSpeechToText()
