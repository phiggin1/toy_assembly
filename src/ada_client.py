#!/usr/bin/env python3

import zmq
import rospy
from toy_assembly.srv import Transcription

class AdaClient:
    def __init__(self):
        rospy.init_node('transcription_service')


        #hostname
        hostname = rospy.get_param("~hostname", "8888")

        #node id
        node_id = rospy.get_param("~node_id", "g03")

        #listening port
        server_port = rospy.get_param("~port", "8888")

        client = context.socket(zmq.REQ)
        client.connect(SERVER_ENDPOINT)
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://*:%s.%s:%s" % (node_id,hostname,server_port))

        self.serv = rospy.Service('get_transciption', Whisper, self.Whisper)
        self.serv = rospy.Service('get_clip_probabilities', CLIP, self.CLIP)
        self.serv = rospy.Service('get_sam_segmentation', SAM, self.SAM)

        rospy.spin()

    def Whisper(self, request):
        rospy.loginfo('transribe req recv')
        self.socket.send_json(request.data)
        transcription = self.socket.recv_json()
        rospy.loginfo('recv from ada')

        return transcription
    
    def CLIP(self, request):
        rospy.loginfo('transribe req recv')
        self.socket.send_json(request.data)
        transcription = self.socket.recv_json()
        rospy.loginfo('recv from ada')

        return transcription
    
    def SAM(self, request):
        rospy.loginfo('transribe req recv')
        self.socket.send_json(request.data)
        transcription = self.socket.recv_json()
        rospy.loginfo('recv from ada')

        return transcription


if __name__ == '__main__':
    get_target = AdaClient()

