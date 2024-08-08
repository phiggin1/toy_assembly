#!/usr/bin/env python3

import rospy
import json
import numpy as np
from toy_assembly.srv import Whisper, WhisperRequest
from toy_assembly.msg import Transcription
from rivr_ros.msg import RivrAudio



class Transcribe:
    def __init__(self):
        rospy.init_node('Transcribe', anonymous=True)


        self.skip_words = [
            "you",
            " you",
            "you ",
            " you ",
            "thank you"
        ]

        self.whisper_serv = rospy.ServiceProxy('get_transciption', Whisper)
        self.transript_publisher = rospy.Publisher("/transcript", Transcription, queue_size=10)
        self.utterance_subscriber = rospy.Subscriber("/utterance", RivrAudio, self.utterance_cb)

        rospy.spin()

    def utterance_cb(self, msg):
        audio = msg.data
        sample_rate = msg.sample_rate

        request  = WhisperRequest()
        audio_data = json.dumps(audio)
        request.string.data = audio_data
        request.sample_rate = sample_rate

        now = rospy.Time.now()
        duration = len(audio)/sample_rate

        #log audio file
        #audio = np.fromstring(audio_data[1:-1], dtype=float, sep=',')

        transcript = self.whisper_serv(request)
        rospy.loginfo(f"transcription: '{transcript.transcription}'")


        for word in self.skip_words:
            if transcript.transcription == word:
                rospy.loginfo(f"skipping word: '{transcript.transcription}'")
                return
            
        t = Transcription()
        t.audio_recieved = now
        t.duration = duration
        t.transcription = transcript.transcription
        #t.audio = audio_data
        
        self.transript_publisher.publish(t)


if __name__ == '__main__':
    t = Transcribe()
