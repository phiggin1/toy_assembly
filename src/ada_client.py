#!/usr/bin/env python3

import zmq
import numpy as np
import rospy
import json
from cv_bridge import CvBridge
from toy_assembly.srv import Whisper, CLIP, SAM
from toy_assembly.srv import WhisperResponse, CLIPResponse, SAMResponse

class AdaClient:
    def __init__(self):
        rospy.init_node('transcription_service')

        self.cvbridge = CvBridge()
        
        server_port = rospy.get_param("~port", "8888")

        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % server_port)
        rospy.loginfo(f"Server listening on port:{server_port}")


        self.whisper_serv = rospy.Service('get_transciption', Whisper, self.Whisper)
        self.clip_serv = rospy.Service('get_clip_probabilities', CLIP, self.CLIP)
        self.sam_serv = rospy.Service('get_sam_segmentation', SAM, self.SAM)

        rospy.spin()

    def Whisper(self, request):
        rospy.loginfo('Whisper req recv')

        audio_json = str(request.data)

        msg = {"type":"whisper",
               "data":audio_json
        }

        self.socket.send_json(msg)
        resp = self.socket.recv_json()

        rospy.loginfo('recv from ada')
        transcription = resp["text"]

        response = WhisperResponse()
        response.transcription = transcription
        return response
    
    def CLIP(self, request):
        rospy.loginfo('CLIP req recv')

        images = []
        for img in request.images:
            images.append(self.cvbridge.imgmsg_to_cv2(img, "bgr8").tolist())
        text = request.text

        msg = {"type":"clip",
               "images":images,
               "text":text
        }

        self.socket.send_json(msg)
        resp = self.socket.recv_json()

        probs = resp["probs"]
        rospy.loginfo('recv from ada')

        response = CLIPResponse()
        response.probs = probs
        return probs
    
    def SAM(self, request):
        rospy.loginfo('SAM req recv')

        image = self.cvbridge.imgmsg_to_cv2(request.image, "bgr8")     
        target_x = request.target_x
        target_y = request.target_y
        
        msg = {"type":"sam",
               "image":image.tolist(),
               "target_x":target_x,
               "target_y":target_y
        }

        rospy.loginfo("sending to ada")
        self.socket.send_json(msg)
        resp = self.socket.recv_json()

        rospy.loginfo('recv from ada')

        masks = []
        for mask in resp["masks"]:
            masks.append(self.cvbridge.cv2_to_imgmsg(np.asarray(mask, dtype=np.uint8)))

        response = SAMResponse()
        response.masks = masks
        return response


if __name__ == '__main__':
    get_target = AdaClient()

