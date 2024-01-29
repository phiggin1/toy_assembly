#!/usr/bin/env python3

import zmq
import json
import rospy
from toy_assembly.srv import Whisper
from toy_assembly.srv import CLIP
from toy_assembly.srv import SAM

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
        rospy.loginfo('Whisper req recv')

        audio_data = request.data

        msg = {"type":"whisper",
               "data":audio_data
        }

        self.socket.send_json(msg)
        resp = self.socket.recv_json()

        rospy.loginfo('recv from ada')
        data = json.dump(resp)
        transcription = data["text"]

        return transcription
    
    def CLIP(self, request):
        rospy.loginfo('CLIP req recv')

        images = request.images
        text = request.text

        msg = {"type":"clip",
               "images":images,
               "text":text
        }


        self.socket.send_json(msg)
        resp = self.socket.recv_json()
        data = json.dump(resp)
        rospy.loginfo('recv from ada')

        return True
    
    def SAM(self, request):
        rospy.loginfo('SAM req recv')

        image = request.image
        target_x = request.target_x
        target_y = request.target_y
        
        msg = {"type":"sam",
               "image":image,
               "target_x":target_x,
               "target_y":target_y
        }

        self.socket.send_json(msg)
        resp = self.socket.recv_json()
        data = json.dump(resp)

        rospy.loginfo('recv from ada')

        masks = data["masks"]

        return masks


if __name__ == '__main__':
    get_target = AdaClient()

