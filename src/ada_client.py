#!/usr/bin/env python3

import zmq
import numpy as np
import rospy
from cv_bridge import CvBridge
from toy_assembly.srv import Whisper, WhisperResponse
from toy_assembly.srv import CLIP, CLIPResponse
from toy_assembly.srv import SAM, SAMResponse
from toy_assembly.srv import TTS, TTSResponse
from toy_assembly.srv import LLMImage, LLMImageResponse 
from toy_assembly.srv import LLMText, LLMTextResponse
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from multiprocessing import Lock

import os
import soundfile as sf

class AdaClient:
    def __init__(self):
        rospy.init_node('ada_services')

        self.mutex = Lock()

        self.cvbridge = CvBridge()
        
        self.debug = rospy.get_param("~debug", True)
        server_port = rospy.get_param("~port", "8888")

        
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % server_port)
        rospy.loginfo(f"Server listening on port:{server_port}")
        
        self.whisper_serv = rospy.Service('/get_transciption', Whisper, self.Whisper)
        #self.clip_serv = rospy.Service('/get_clip_probabilities', CLIP, self.CLIP)
        self.sam_serv = rospy.Service('/get_sam_segmentation', SAM, self.SAM)
        #self.tts_serv = rospy.Service("/get_text_to_speech", TTS, self.TTS)
        
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
    
    def CLIP(self, request):
        if self.debug: rospy.loginfo('CLIP req recv')

        images = []
        for img in request.images:
            images.append(self.cvbridge.imgmsg_to_cv2(img, "bgr8").tolist())
        text = request.text

        msg = {"type":"clip",
               "images":images,
               "text":text
        }

        with self.mutex:
            self.socket.send_json(msg)
            resp = self.socket.recv_json()

        probs = np.asarray(resp["probs"])
        if self.debug: rospy.loginfo('CLIP recv from ada')
        rospy.loginfo(f"CLIP probs:{probs}")

        response = CLIPResponse()
        resp_probs = Float32MultiArray()
        resp_probs.layout.dim = [MultiArrayDimension('dim%d' % i,  probs.shape[i], probs.shape[i] * probs.dtype.itemsize) for i in range(probs.ndim)]
        resp_probs.data = probs.reshape([1, -1])[0].tolist()
        response.probs = resp_probs

        return response
    
    def SAM(self, request):
        if self.debug: rospy.loginfo('SAM req recv')

        image = self.cvbridge.imgmsg_to_cv2(request.image, "bgr8")     
        target_x = request.target_x
        target_y = request.target_y
        
        print(image.shape)

        msg = {"type":"sam",
               "image":image.tolist(),
               "target_x":target_x,
               "target_y":target_y
        }

        if self.debug: rospy.loginfo("SAM sending to ada")
        
        with self.mutex:
            self.socket.send_json(msg)
            resp = self.socket.recv_json()
        if self.debug: rospy.loginfo('SAM recv from ada') 

        rospy.loginfo(resp["scores"])


        masks = []
        for mask in resp["masks"]:
            m = np.asarray(mask, dtype=np.uint8)*255
            masks.append(self.cvbridge.cv2_to_imgmsg(m))

        response = SAMResponse()
        response.masks = masks
        return response

    def TTS(self, request):
        if self.debug: rospy.loginfo('TTS req recv')

        text = request.text
        print(request)
        print(text)
        msg = {"type":"tts",
               "text":text
        }

        if self.debug: rospy.loginfo("TTS sending to ada")
        if self.debug: rospy.loginfo(f"Text:{text}")
        with self.mutex:
            self.socket.send_json(msg)
            resp = self.socket.recv_json()
        if self.debug: rospy.loginfo('TTS recv from ada') 

        text = resp["text"]
        rate = resp["rate"]
        audio = resp["audio"]
        rospy.loginfo(f"rate:{rate}, text:{text}")

        audio=  np.fromstring(audio[1:-1], dtype=float, sep=',')
        float_array = Float32MultiArray()
        float_array.data = audio

        resp = TTSResponse()
        resp.audio = float_array
        return resp
    
    def LLMImage(self, request):
        if self.debug: rospy.loginfo('LLMImage req recv')

        image = self.cvbridge.imgmsg_to_cv2(request.image, "bgr8")      
        print(image.shape)

        msg = {"type":"llm_image",
               "image":image.tolist(),
        }

        if self.debug: rospy.loginfo("LLMImage sending to ada")
        
        with self.mutex:
            self.llm_socket.send_json(msg)
            resp = self.llm_socket.recv_json()
        if self.debug: rospy.loginfo('LLMImage recv from ada') 

        #TODO Add in reponse
        resp = LLMImageResponse()
        resp.result = True

        return resp

    def LLMText(self, request):
        if self.debug: rospy.loginfo('LLMText req recv')

        text = request.text
        print(request)
        print(text)
        msg = {"type":"llm_ask",
               "text":text
        }

        if self.debug: rospy.loginfo("LLMText sending to ada")
        if self.debug: rospy.loginfo(f"Text:{text}")
        with self.mutex:
            self.llm_socket.send_json(msg)
            resp = self.llm_socket.recv_json()
        if self.debug: rospy.loginfo('LLMText recv from ada') 

        text = resp["text"]
        rospy.loginfo(f"text:{text}")

        resp = LLMTextResponse()
        resp.text = text

        return resp

if __name__ == '__main__':
    get_target = AdaClient()

