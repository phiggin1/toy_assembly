#!/usr/bin/env python3

import zmq
import numpy as np
import rospy
from cv_bridge import CvBridge
from toy_assembly.srv import SAM, SAMResponse
from multiprocessing import Lock

import os
import soundfile as sf

class AdaClient:
    def __init__(self):
        rospy.init_node('ada_sam')

        self.mutex = Lock()

        self.cvbridge = CvBridge()
        
        self.debug = rospy.get_param("~debug", True)
        server_port = rospy.get_param("~port", "8899")
        
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % server_port)
        rospy.loginfo(f"Server listening on port:{server_port}")
        
        self.sam_serv = rospy.Service('/get_sam_segmentation', SAM, self.SAM)
        
        rospy.spin()

    
    def SAM(self, request):
        if self.debug: rospy.loginfo('SAM req recv')

        image = self.cvbridge.imgmsg_to_cv2(request.image, "bgr8")     
        text = "tan tray. orange tray. horse body. blue horse legs. orange horse legs."

        print(image.shape)

        msg = {"type":"sam",
               "image":image.tolist(),
               "text":text,
        }

        if self.debug: rospy.loginfo("SAM sending to ada")
        
        with self.mutex:
            self.socket.send_json(msg)
            resp = self.socket.recv_json()
        if self.debug: rospy.loginfo('SAM recv from ada') 

        rospy.loginfo(resp)


        '''
        masks = []
        for mask in resp["masks"]:
            m = np.asarray(mask, dtype=np.uint8)*255
            masks.append(self.cvbridge.cv2_to_imgmsg(m))
        '''
        response = SAMResponse()
        #response.masks = masks
        return response

if __name__ == '__main__':
    get_target = AdaClient()

