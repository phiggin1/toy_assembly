#!/usr/bin/env python3

import rospy
import json
import numpy as np
import io
from std_msgs.msg import String
from toy_assembly.srv import TTS
from toy_assembly.srv import TTSRequest
from std_msgs.msg import Float32MultiArray
import sounddevice

class RobotSpeech:
    def __init__(self):
        rospy.init_node('robot_speech')
        self.sample_rate = rospy.get_param("~sample_rate", 22050)
        self.speech_delay = rospy.get_param("~speech_delay", 0.0)
        self.last_time_spoke = None
        self.rivr_robot_speech = rospy.Publisher('/robotspeech', Float32MultiArray, queue_size=10)
        self.tts_sub = rospy.Subscriber('/text_to_speech', String, self.talk)

        rospy.wait_for_service('get_text_to_speech')
        self.tts_serv = rospy.ServiceProxy('get_text_to_speech', TTS)

        rospy.spin()
    
    def talk(self, str_msg):
        text = str_msg.data
        rospy.loginfo(f"text:{text}")
        now  = rospy.Time.now().to_sec()
        if (self.last_time_spoke is None or now > (self.last_time_spoke+self.speech_delay)):
            self.last_time_spoke = now
            req = TTSRequest()
            req.text = text
            resp = self.tts_serv(req)
            float_audio_array = resp.audio

            self.rivr_robot_speech.publish(float_audio_array)

            audio = np.asarray(float_audio_array.data)
            duration = float(audio.shape[0]/self.sample_rate)

            sounddevice.play(audio, self.sample_rate)  
            rospy.sleep(duration)

if __name__ == '__main__':
    tts = RobotSpeech()