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

        self.messages = []

        self.llm_serv = rospy.Service("/gpt_servcice_check", LLMImage, self.call_gpt_check)
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

        self.get_prompt(text, image, objects)
        ans = self.chat_complete(self.messages)
        
        self.prev_answer = ans

        return ans
    
    def call_gpt_check(self, req):
        text = req.text
        check = req.check
        image = req.image
        objects = req.objects

        if self.debug: rospy.loginfo("============================")
        if self.debug: rospy.loginfo(f"call_gpt_check transcript:{text}, check:{check}, objects:{objects}")

        self.get_check_prompt(text, image, objects)
        ans = self.chat_complete(self.messages)
        
        self.prev_answer = ans

        return ans

    def get_prompt(self, text, image, objects):
        print("get_prompt")
        cv_img = self.cvbridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        is_success, buffer = cv2.imencode(".png", cv_img)
        io_buf = io.BytesIO(buffer)        

        encoded_image = base64.b64encode(buffer).decode("utf-8") 

        system = """
You are an excellent interpreter of human instructions for basic tasks. 
You are working with a human to jointly perform a simple collaborative task. 
In this task you are a robot working with a human to build a slot together toy.

Please do not begin working until I say "Start working." 
Instead, simply ouput the message "Waiting for next input." 
Understood?
"""
        prompt = """
Start working. 
For a given statement determine what action the robot should take.
If the "PICKUP" action is chosen there must be a "object" key in the returned dictonary with the object from the list below as the value.
Return only a single object from the list of objects provided.
Resume using the following instruction and the objects in the provided image.

---
The instruction is as follows:
---
{"instruction": '[INSTRUCTION]}
---
{"objects" = [OBJECTS]}
{"actions" = ["MOVE_RIGHT", "MOVE_LEFT", "MOVE_UP", "MOVE_DOWN", "MOVE_FORWARD", "MOVE_BACKWARD", "TILT_UP", "TILT_DOWN", "ROTATE_LEFT", "ROTATE_RIGHT", "PICKUP", "OPEN_HAND", "CLOSE_HAND", "OTHER"]}
---
The dictonary that you return should be formatted as python dictonary. Follow these rules:
1. Never leave ',' at the end of the list.
2. All keys of the dictionary should be double-quoted.
3. Insert ``` at the beginning and the end of the dictionary to separate it from the rest of your response.
"""
        instruction = deepcopy(prompt)
        if instruction.find('[INSTRUCTION]') != -1:
            instruction = prompt.replace('[INSTRUCTION]', text)
        if instruction.find('[OBJECTS]') != -1:
            instruction = instruction.replace('[OBJECTS]', ", ".join(objects))

        self.messages = [
            {
                "role":"system", 
                "content": [
                    {"type":"text", "text" : system},
                ]
            },
            {
                "role":"user", 
                "content": [
                    {"type":"text", "text" : instruction},
                    {"type":"image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}#, "detail":"low" }}
                ]
            }
        ] 
        
    def get_check_prompt(self, text, image, objects):
        print("get_check_prompt")
        check_prompt = """
Start working. 
For a given statement determine if the previous selection was correct and put it into a python dictionary. 

The dictionary has two keys.
---
- dictonary["correct"] : "True" if the statement verifies the "previous_selection" was correct otherwise "False".
- dictonary["oject"] : If not correct give an updated guess, using the previous instruction and the new instuction.
---
Resume using the following instruction and the objects in the provided image.
---
The instruction is as follows:
---
{"instruction": '[INSTRUCTION]
"objects" = [OBJECTS],
"previous_selection" = [PREVIOUS_SELECTION]
}
---
The dictonary that you return should be formatted as python dictonary. Follow these rules:
1. Never leave ',' at the end of the list.
2. All keys of the dictionary should be double-quoted.
3. Insert ``` at the beginning and the end of the dictionary to separate it from the rest of your response.
3. Only use the objects listed in the environment.
"""
        check_text = text
        prev_inst = self.prev_answer
        check_instruction = deepcopy(check_prompt)
        if check_instruction.find('[INSTRUCTION]') != -1:
            check_instruction = check_instruction.replace('[INSTRUCTION]', check_text)
        if check_instruction.find('[PREVIOUS_SELECTION]') != -1:
            check_instruction = check_instruction.replace('[PREVIOUS_SELECTION]', prev_inst)
        if check_instruction.find('[OBJECTS]') != -1:
            check_instruction = check_instruction.replace('[OBJECTS]', ", ".join(objects))

        self.messages.append(            {
                "role":"user", 
                "content": [
                    {"type":"text", "text" : check_instruction},
                ]
            }
        )

    def chat_complete(self, messages):
        start_time = time.time_ns()

        model = "gpt-4o-2024-05-13"
        #model = "gpt-4o-mini-2024-07-18"
        temperature = 0.0
        max_tokens = 128
        
        results = self.client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        response = []
        for chunk in results:
            response.append(chunk.choices[0].delta.content)

        ans = [m for m in response if m is not None]
        answer = ''.join([m for m in ans])
        '''
        answer = self.woz(messages)
        '''
        end_time = time.time_ns()

        rospy.loginfo(f"GPT resp:\n {answer} \n latency: {(end_time-start_time)/(10**9)}")

        return answer
    
    def woz(self, messages):
        n_msgs = len(messages)
        print(n_msgs)

        print("=================")
        if n_msgs > 2:
            correct = input("Correct (y/n): ")
            if correct == "y":
                correct = True
            else:
                correct = False
            obj = input(f"Object # (0-2): ")
            answer = f"""'''{{"correct":"{correct}","object":"obj_{obj}"}}'''"""
        else:
            obj = input("Object # (0,1,2): ")
            answer = f"""'''{{"object":"obj_{obj}"}}'''"""
        print("=================")

        return answer
        

if __name__ == '__main__':
    llm = LLMClient()
