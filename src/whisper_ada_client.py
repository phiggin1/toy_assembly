#!/usr/bin/env python3

import zmq
import numpy as np
import rospy
from toy_assembly.srv import Whisper, WhisperResponse
from multiprocessing import Lock
import os
import soundfile as sf

class AdaClient:
    def __init__(self):
        rospy.init_node('whisper_ada_services')

        self.mutex = Lock()
        
        self.debug = rospy.get_param("~debug", True)
        server_port = rospy.get_param("~port", "8888")

        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % server_port)
        rospy.loginfo(f"Server listening on port:{server_port}")
        
        self.whisper_serv = rospy.Service('/get_transciption', Whisper, self.Whisper)
        
        rospy.spin()

    def Whisper(self, request):
        if self.debug: rospy.loginfo('Whisper req recv')
        sample_rate = request.sample_rate

        now = rospy.Time.now().nsecs
        tmp_audio_filename = os.path.join("/home/rivr/audio_test", f"{now}.wav")
        audio = np.fromstring(request.string.data[1:-1], dtype=float, sep=',')

        sf.write(tmp_audio_filename, audio, sample_rate)

        audio_json = str(request.string.data)
        context = ""


        msg = {"type":"whisper",
               "context":context,
               "sample_rate":sample_rate,
               "data":audio_json
        }

        with self.mutex:
            self.socket.send_json(msg)
            resp = self.socket.recv_json()

        if self.debug: rospy.loginfo('Whisper recv from ada')
        

        #print(resp)
        transcription = resp["text"]
        
        rospy.loginfo(f"Whisper transcription: '{transcription}'")

        response = WhisperResponse()
        response.transcription = transcription
        return response
    
if __name__ == '__main__':
    get_target = AdaClient()

