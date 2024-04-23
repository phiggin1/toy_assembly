#!/usr/bin/env python3

import rospy
import json
import numpy as np
import io
from std_msgs.msg import String
from toy_assembly.srv import Whisper
from toy_assembly.srv import WhisperRequest
from toy_assembly.msg import Transcription
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension

from scipy.io.wavfile import write as wavfile_writer

def is_silent(snd_data, threshold):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < threshold


class AudioSpeechToText:
    def __init__(self):
        rospy.init_node('AudioSpeechToText', anonymous=True)

        self.debug = rospy.get_param("~debug", True) 

        #Number of audio messages that are below threshold 
        #   to determine if person stopped talking
        self.silent_wait = rospy.get_param("~silent_wait", 3)

        #Maximum audio clip length (seconds)
        self.max_duration = rospy.get_param("~max_duration", 5)

        #Threshold to detect when there is sound 
        # normalized ([0,1.0])
        self.threshold = rospy.get_param("~threshold", 0.01)
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

        #self.robot_statments = rospy.Subscriber("robot_spp")

        rospy.wait_for_service('get_transciption')        
        self.whisper_serv = rospy.ServiceProxy('get_transciption', Whisper)
        self.audio_subscriber = rospy.Subscriber("/audio", Float32MultiArray, self.audio_cb)
        self.transript_publisher = rospy.Publisher("/transcript", Transcription, queue_size=10)

        rospy.spin()
    
    def audio_cb(self, msg):
        float_array = msg.data
        if self.debug: rospy.loginfo(f"msg recv, max volumn:{max(float_array)}")
        self.process_audio(float_array)
    
    def process_audio(self, data):
        silent = is_silent(data, self.threshold)
        if not silent:
            if self.debug: rospy.loginfo("there is sound")

        if not silent and not self.snd_started:
            self.snd_started = True
            self.num_silent = 0
            self.audio_clip = []
        elif silent and self.snd_started:
            self.num_silent += 1
        elif not silent and self.snd_started:
            self.num_silent = 0           
            rospy.loginfo(f"num_silent:{self.num_silent}")


        if self.snd_started:
            self.audio_clip.extend(data)

        if self.snd_started and self.num_silent > self.silent_wait:     #enough quite time that they stopped speaking
            if self.debug: rospy.loginfo(f"got audio clip, num_silent:{self.num_silent}")
            self.get_transcription(self.audio_clip)
            self.snd_started = False
            self.num_silent = 0
            self.audio_clip = []
        elif len(self.audio_clip)>(self.sample_rate*self.max_duration) and self.snd_started:    #hit the maxiumum clip duration
            if self.debug: rospy.loginfo("max clip length")
            self.get_transcription(self.audio_clip)
            self.snd_started = False
            self.num_silent = 0
            self.audio_clip = []

    def get_transcription(self, audio):
            request  = WhisperRequest()
            audio_data = json.dumps(audio)
            request.string.data = audio_data

            now = rospy.Time.now()
            duration = len(audio)/self.sample_rate

            #log audio file
            #audio = np.fromstring(audio_data[1:-1], dtype=float, sep=',')
            #wavfile_writer("/home/phiggin1/"+str(rospy.time.Now().to_sec())+".wav", self.sample_rate, audio)
    
            transcript = self.whisper_serv(request)
            rospy.loginfo(f"get_transcription:{transcript.transcription}")
            #publish full audio message (wavbytes and text)

            t = Transcription()
            t.audio_recieved = now
            t.duration = duration
            t.transcription = transcript.transcription
            #t.audio = audio_data
            self.transript_publisher.publish(t)
            

            self.snd_started = False
            self.num_silent = 0
            self.audio_clip = []


if __name__ == '__main__':
    get_target = AudioSpeechToText()
