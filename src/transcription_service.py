#!/usr/bin/env python3

import zmq
import rospy
from toy_assembly.srv import Transcription

class TranscribeClient:
    def __init__(self):
        rospy.init_node('transcription_service')

        sever_port  = "8888"
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % sever_port)

        self.serv = rospy.Service('get_transciption', Transcription, self.transribe)
        rospy.spin()

    def transribe(self, request):
        self.socket.send(request.data)
        transcription = self.socket.recv_string()
        #transcription = "test"
        return transcription


if __name__ == '__main__':
    get_target = TranscribeClient()

