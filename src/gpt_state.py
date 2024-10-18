#!/usr/bin/env python3

import io
import cv2
import time
import json
from copy import deepcopy
from openai import OpenAI
import base64
import rospy
from cv_bridge import CvBridge
from toy_assembly.srv import LLMImage, LLMImageRequest, LLMImageResponse

class LLMClient:
    def __init__(self):
        rospy.init_node('GPTCheckState')

        self.debug = rospy.get_param("~debug", True)

        self.cvbridge = CvBridge()

        key_filename = rospy.get_param("~key_file", "/home/phiggin1/ai.key")
        with open(key_filename, "rb") as key_file:
            key = key_file.read().decode("utf-8")
            
        self.client = OpenAI(
            api_key = key,
        )
        

        fp_system = "/home/rivr/toy_ws/src/toy_assembly/prompts/gpt_state_system.txt"
        with open(fp_system) as f:
            self.system = f.read()

        fp_state = "/home/rivr/toy_ws/src/toy_assembly/prompts/gpt_env_state.txt"
        with open(fp_state) as f:
            self.state = f.read()

        fp_prompt = "/home/rivr/toy_ws/src/toy_assembly/prompts/gpt_state_prompt.txt"
        with open(fp_prompt) as f:
            self.prompt = f.read()


        self.messages = [
            {
                "role":"system", 
                "content": 
                [ {"type":"text", "text" : self.system} ]
            },
            {
                "role":"user", 
                "content": 
                [ {"type":"text", "text" : self.state} ]
            },
            {
                "role":"assistant", 
                "content": [ {"type":"text", "text" : "Understood. Waiting for next input."} ]
            }
        ] 

        self.llm_serv = rospy.Service("/gpt_state_servcice", LLMImage, self.call_gpt)

        rospy.spin()

    def call_gpt(self, req):
        rospy.loginfo("recv req")
        #text will be the results of the action

        text = req.text
        image = req.image
        objects = req.objects
        env = req.env

        if self.debug: rospy.loginfo("============================")
        if self.debug: print(f"Sending to gpt state\ntext:{text}")

        self.messages = [
            {
                "role":"system", 
                "content": 
                [ {"type":"text", "text" : self.system} ]
            },
            {
                "role":"user", 
                "content": 
                [ {"type":"text", "text" : self.state} ]
            },
            {
                "role":"assistant", 
                "content": [ {"type":"text", "text" : "Understood. Waiting for next input."} ]
            }
        ] 
        new_msg = self.get_prompt(text, image, objects, env)
        ans = self.chat_complete(new_msg)
        
        resp  = LLMImageResponse()
        resp.text = ans

        return resp


    def get_prompt(self, text, image, objects, env):
        print("get_prompt")
        json_dict = json.loads(text)
        action = json_dict["action"]
        success = json_dict["success"]

        cv_img = self.cvbridge.imgmsg_to_cv2(image, desired_encoding="rgb8")
        is_success, buffer = cv2.imencode(".png", cv_img)
        io_buf = io.BytesIO(buffer)        
        encoded_image = base64.b64encode(buffer).decode("utf-8") 

        instruction = deepcopy(self.prompt)
        if instruction.find('[ACTION]') != -1:
            instruction = instruction.replace('[ACTION]', str(action))
        if instruction.find('[SUCCESS]') != -1:
            instruction = instruction.replace('[SUCCESS]', str(success))
        if instruction.find('[OBJECTS]') != -1:
            instruction = instruction.replace('[OBJECTS]', ", ".join(objects))
        if instruction.find('[ENVIRONMENT]') != -1:
            instruction = instruction.replace('[ENVIRONMENT]', env)

        prompt_dict = {
            "role":"user", 
            "content": 
            [
                {"type":"text", "text" : instruction},
                {"type":"image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }

        return prompt_dict

    def chat_complete(self, new_msg):
        start_time = time.time_ns()
        
        self.messages.append(new_msg)

        model = "gpt-4o-2024-05-13"
        #model = "gpt-4o-mini-2024-07-18"
        temperature = 0.0
        max_tokens = 350
        
        results = self.client.chat.completions.create(model=model, messages=self.messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        response = []
        for chunk in results:
            response.append(chunk.choices[0].delta.content)

        ans = [m for m in response if m is not None]
        answer = ''.join([m for m in ans])
        
        self.messages.append(
            {
                "role":"assistant", 
                "content": [
                    {"type":"text", "text" : answer},
                ]
            }
        )

        # only keep the system, and the last message
        if len(self.messages) > 4:
            del self.messages[1:2]

        end_time = time.time_ns()

        one_line_asn = str(answer).replace('\n','')
        rospy.loginfo(f"GPT resp:\n {one_line_asn}")
        rospy.loginfo(f"latency: {(end_time-start_time)/(10**9)}")

        return answer
    

        

if __name__ == '__main__':
    llm = LLMClient()
