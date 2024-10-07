#!/usr/bin/env python3

import io
import cv2
import time
from copy import deepcopy
from openai import OpenAI
import base64
import rospy
import tf
from cv_bridge import CvBridge
from toy_assembly.srv import LLMImage

class LLMClient:
    def __init__(self):
        rospy.init_node('LLM_testing')

        self.debug = rospy.get_param("~debug", True)

        self.cvbridge = CvBridge()
        self.listener = tf.TransformListener()

        key_filename = rospy.get_param("~key_file", "/home/phiggin1/ai.key")
        with open(key_filename, "rb") as key_file:
            key = key_file.read().decode("utf-8")
            
        self.client = OpenAI(
            api_key = key,
        )

        fp_actions = "/home/rivr/toy_ws/src/toy_assembly/prompts/actions.txt"
        with open(fp_actions) as f:
            self.actions = f.read()

        fp_system = "/home/rivr/toy_ws/src/toy_assembly/prompts/gpt_system.txt"
        with open(fp_system) as f:
            self.system = f.read()
        if self.system.find("[ACTIONS]") != -1:
            self.system = self.system.replace("[ACTIONS]", self.actions)
        print(self.system)

        fp_prompt = "/home/rivr/toy_ws/src/toy_assembly/prompts/gpt_prompt.txt"
        with open(fp_prompt) as f:
            self.prompt = f.read()
        if self.prompt.find("[ACTIONS]") != -1:
            self.prompt = self.prompt.replace("[ACTIONS]", self.actions)
        print(self.prompt)
    
        self.messages = [
            {
                "role":"system", 
                "content": [
                    {"type":"text", "text" : self.system},
                ]
            },
        ] 

        self.llm_check_serv = rospy.Service("/gpt_servcice", LLMImage, self.call_gpt)

        rospy.spin()

    def call_gpt(self, req):
        text = req.text
        check = req.check
        image = req.image
        objects = req.objects
        rospy.loginfo(f"call_gpt transcript:{text}, check:{check}, objects:{objects}")

        if self.debug: rospy.loginfo("============================")
        if self.debug: print(f"Sending to gpt\ntext:{text}")

        self.messages = [
            {
                "role":"system", 
                "content": [
                    {"type":"text", "text" : self.system},
                ]
            },
        ] 
        
        new_msg = self.get_prompt(text, image, objects)
        ans = self.chat_complete(new_msg)
        
        self.prev_answer = ans

        return ans


    def get_prompt(self, text, image, objects):
        print("get_prompt")
        cv_img = self.cvbridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        merge_test = "_".join(text.split(" "))
        fname = f"/home/rivr/toy_logs/images/{image.header.stamp}{merge_test}.png"
        print(fname)
        cv2.imwrite(fname, cv_img)
        is_success, buffer = cv2.imencode(".png", cv_img)
        io_buf = io.BytesIO(buffer)        
        encoded_image = base64.b64encode(buffer).decode("utf-8") 
 
        instruction = deepcopy(self.prompt)
        if instruction.find('[INSTRUCTION]') != -1:
            instruction = instruction.replace('[INSTRUCTION]', text)
        if instruction.find('[OBJECTS]') != -1:
            instruction = instruction.replace('[OBJECTS]', ", ".join(objects))

        return {"role":"user", 
                "content": [
                    {"type":"text", "text" : instruction},
                    {"type":"image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]}

    def chat_complete(self, new_msg):
        start_time = time.time_ns()
        
        self.messages.append(new_msg)

        model = "gpt-4o-2024-05-13"
        #model = "gpt-4o-mini-2024-07-18"
        temperature = 0.0
        max_tokens = 128
        
        results = self.client.chat.completions.create(model=model, messages=self.messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        response = []
        for chunk in results:
            response.append(chunk.choices[0].delta.content)

        ans = [m for m in response if m is not None]
        answer = ''.join([m for m in ans])
        
        #answer = self.woz(self.messages)
        
        
        self.messages.append(
            {
                "role":"assistant", 
                "content": [
                    {"type":"text", "text" : answer},
                ]
            }
        )

        # only keep the system, and the last message
        if len(self.messages) > 3:
            del self.messages[1:2]

        end_time = time.time_ns()

        rospy.loginfo(f"GPT resp:\n {answer} \n latency: {(end_time-start_time)/(10**9)}")

        return answer
    
    def woz(self, messages):
        n_msgs = len(messages)
        print(n_msgs)

        print("=================")

        action = input("action: ")
        obj = input(f"Object # (0-2): ")
        answer = f"""'''{{"action":"{action}","object":"obj_{obj}"}}'''"""
        rospy.loginfo(answer)

        print("=================")

        return answer
        

if __name__ == '__main__':
    llm = LLMClient()
