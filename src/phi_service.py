#!/usr/bin/env python3

import zmq
import rospy
from toy_assembly.srv import LLMText, LLMTextRequest, LLMTextResponse

class LLMClient:
    def __init__(self):
        rospy.init_node('PHI_LLM')

        self.debug = rospy.get_param("~debug", True)
        server_port = rospy.get_param("~port", "8877")

        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % server_port)
        rospy.loginfo(f"Server listening on port:{server_port}")


        self.llm_serv = rospy.Service("/phi_servcice", LLMText, self.call_phi)

        rospy.spin()

    def call_phi(self, req):
        text = req.text
        #env = req.env
        rospy.loginfo(f"call_phi text:{text}")

        msg = {"type":"llm",
               "text":text,
               "prev":""
        }

        if self.debug: rospy.loginfo(f"LLM sending to ada\ntext:{text}")
        
        self.socket.send_json(msg)
        resp = self.socket.recv_json()


        phi_resp = LLMTextResponse()
        phi_resp.text = resp["text"]

        return phi_resp
        

if __name__ == '__main__':
    llm = LLMClient()
