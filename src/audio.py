#!/usr/bin/env python3

import rospy
import json
import numpy as np
import io
from std_msgs.msg import String
from toy_assembly.srv import Transcription
from scipy.io.wavfile import write

def is_silent(snd_data, threshold):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < threshold


class AudioSpeechToText:
    def __init__(self):
        rospy.init_node('AudioSpeechToText', anonymous=True)

        self.debug = rospy.get_param("~debug", True) 

        #Number of audio messages that are below threshold 
        #   to determine if person stopped talking
        self.silent_wait = rospy.get_param("~silent_wait", 10)

        #Maximum audio clip length (seconds)
        self.max_duration = rospy.get_param("~max_duration", 25)

        #Threshold to detect when there is sound 
        # normalized ([0,1.0])
        self.threshold = rospy.get_param("~", 0.1)

        #Audio sample rate (hz)
        self.sample_rate = rospy.get_param("~sample_rate", 16000)

        #counter for number of silent audio messages
        self.num_silent = 0
        self.snd_started = False
        self.audio_clip = []

        rospy.wait_for_service('get_transciption')        
        self.serv = rospy.ServiceProxy('get_transciption', Transcription)
        self.audio_subscriber = rospy.Subscriber("/audio", String, self.audio_cb)

        rospy.spin()
    
    def audio_cb(self, msg):
        rospy.loginfo("msg recv")
        float_array = json.loads(msg.data)
        self.process_audio(float_array)
    

    def process_audio(self, data):
        silent = is_silent(data, self.threshold)

        if not silent:
            rospy.loginfo("there is sound")

        self.audio_clip.extend(data)

        if silent and self.snd_started:
            self.num_silent += 1
        elif not silent and not self.snd_started:
            self.snd_started = True
            self.num_silent = 0
            self.audio_clip = []

        if self.snd_started and self.num_silent > self.silent_wait:
            rospy.loginfo("got audio clip")
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
            rospy.loginfo("max clip length")
            self.get_transcription(self.audio_clip)

    def get_transcription(self, audio):
            #get audio into correct format (binary wav file)
            data = np.asarray(audio)
            bytes_wav = bytes()
            byte_io = io.BytesIO(bytes_wav)
            write(byte_io, self.sample_rate, data)
            wav_data = byte_io.read()

            #get the transcription here
            print(data[0])
            print(wav_data[0])
            transcript = self.serv(wav_data)
            print(transcript.transcription)
            #publish full audio message (wavbytes and text)

            self.snd_started = False
            self.num_silent = 0
            self.audio_clip = []


if __name__ == '__main__':
    get_target = AudioSpeechToText()
