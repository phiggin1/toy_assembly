#!/usr/bin/env python3

import rospy
import json
import numpy as np
import io
from std_msgs.msg import String
from toy_assembly.srv import Whisper
from toy_assembly.srv import WhisperRequest
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension


def is_silent(snd_data, threshold):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < threshold


class AudioSpeechToText:
    def __init__(self):
        rospy.init_node('AudioSpeechToText', anonymous=True)

        self.debug = rospy.get_param("~debug", True) 

        #Number of audio messages that are below threshold 
        #   to determine if person stopped talking
        self.silent_wait = rospy.get_param("~silent_wait", 5)

        #Maximum audio clip length (seconds)
        self.max_duration = rospy.get_param("~max_duration", 5)

        #Threshold to detect when there is sound 
        # normalized ([0,1.0])
        self.threshold = rospy.get_param("~threshold", 0.005)
        if self.threshold < 0.0:
            rospy.loginfo("threshold should be normalized ([0,1.0])")
            self.threshold = 0.0
        elif self.threshold > 1.0:
            rospy.loginfo("threshold should be normalized ([0,1.0])")
            self.threshold = 1.0

        #Audio sample rate (hz)
        self.sample_rate = rospy.get_param("~sample_rate", 16000)

        rospy.loginfo("debug: %r"%self.debug)
        rospy.loginfo("silent_wait: %i"%self.silent_wait )
        rospy.loginfo("max_duration: %i"%self.max_duration )
        rospy.loginfo("threshold: %f"%self.threshold )
        rospy.loginfo("sample_rate: %i"%self.sample_rate )

        #counter for number of silent audio messages
        self.num_silent = 0
        self.snd_started = False
        self.audio_clip = []

        rospy.wait_for_service('get_transciption')        
        self.whisper_serv = rospy.ServiceProxy('get_transciption', Whisper)
        self.audio_subscriber = rospy.Subscriber("/audio", String, self.audio_cb)
        self.transript_publisher = rospy.Publisher("/transcript", String, queue_size=10)

        rospy.spin()
    
    def audio_cb(self, msg):
        float_array = json.loads(msg.data)
        if self.debug: rospy.loginfo(f"msg recv, max volumn:{max(float_array)}")
        self.process_audio(float_array)
    
    def process_audio(self, data):

        silent = is_silent(data, self.threshold)
        if not silent:
            if self.debug: rospy.loginfo("there is sound")

        self.audio_clip.extend(data)

        if silent and self.snd_started:
            self.num_silent += 1
        elif not silent and not self.snd_started:
            self.snd_started = True
            self.num_silent = 0
            self.audio_clip = []

        if self.snd_started and self.num_silent > self.silent_wait:
            if self.debug: rospy.loginfo("got audio clip")
            self.get_transcription(self.audio_clip)
        
        '''
        add in some code for potential audio recording visualization aides
        if snd_started and not silent -> ?green activly lisenting
        elif snd_started and num_silent > 0(some number less than silent_wait greater than zero) -> ?yellow thinks speaker is finished but unsure
        else -> ?red waiting for someone to talk
        '''

        #if the audio clip is over max duration get the transcription
        #clip is len(audio_clip)/rate seconds long
        if len(self.audio_clip)>(self.sample_rate*self.max_duration) and self.snd_started:
            if self.debug: rospy.loginfo("max clip length")
            self.get_transcription(self.audio_clip)

    def get_transcription(self, audio):
            request  = WhisperRequest()
            request.data.data = json.dumps(audio)

            transcript = self.whisper_serv(request)
            rospy.loginfo(f"get_transcription:{transcript.transcription}")
            #publish full audio message (wavbytes and text)
            self.transript_publisher.publish(transcript.transcription)

            self.snd_started = False
            self.num_silent = 0
            self.audio_clip = []


if __name__ == '__main__':
    get_target = AudioSpeechToText()
