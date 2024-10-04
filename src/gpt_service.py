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


        self.system = """
You are an excellent interpreter of human instructions for basic tasks. 
You are working with a human to jointly perform a simple collaborative task. 
In this task you are a robot working with a human to build a slot together toy.
You are controlling a second robot arm that may be seen in the images.

Please do not begin working until I say "Start working." 
Instead, simply ouput the message "Waiting for next input." 
Understood?
"""
        actions_list = [
          ["MOVE_FORWARD", "Move the robot's hand forward toward the human"], 
          ["MOVE_BACKWARD", "Move the robot's hand backward away from the human"], 
          ["MOVE_RIGHT", "Move the robot's hand to the robot's right, the human's left"], 
          ["MOVE_LEFT", "Move the robot's hand to the robot's left, the human's right"],
          ["MOVE_UP", "Move the robot's hand up."], 
          ["MOVE_DOWN", "Move the robot's hand down."], 
          ["PITCH_UP", "Tilt the robot's hand up."],
          ["PITCH_DOWN", "Tilt the robot's hand down."],
          ["ROLL_LEFT", "Rotate the robot's hand to the left."],
          ["ROLL_RIGHT", "Rotate the robot's hand to the right."],
          ["OPEN_HAND", "Opens the robots hand, lets go of any held object."], 
          ["CLOSE_HAND", "Close the robots hand, grab objects between the robots fingers"],
          ["PICK_UP", "Move the arm to pick up an object"],
          ["OTHER", "Any other possible command"]
        ]
        self.actions = ""
        for a in actions_list:
          self.actions += " - "+a[0]+" : " + a[1]+"\n"

        self.system = """
You are an excellent interpreter of human instructions for basic tasks. 
You are working with a human to jointly perform a simple collaborative task. 
In this task you are a robot working with a human to build a slot together toy.

actions = {[ACTIONS]}

For a given statement determine if the statement is directed to the robot, is not a request or is not action.
If it is not return the action the robot should as a python dictonary.
The dictionary has one key.
---
- dictonary["action"] : A list of the actions that the robot should take, taken from the actions list above.
---
-------------------------------------------------------
"""
        if self.system.find("[ACTIONS]") != -1:
            self.system = self.system.replace("[ACTIONS]", self.actions)


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

        prompt = """
Start working. 
For a given statement determine what actions the robot should take.
If the "PICKUP" action is chosen there must be a "object" key in the returned dictonary with the object from the list below as the value.
Return only a single object from the list of objects provided.
You should only choose "PICKUP" if the person instructs and if there are any objects in the "objects' list.
If the "MOVE_TO" action is chosen there must either an "object" or "direction" in the returned dictonary. 
If "object" is used it must be from the list below as the value.
If direction is used it must be one of "MOVE_RIGHT", "MOVE_LEFT", "MOVE_UP", "MOVE_DOWN", "MOVE_FORWARD", "MOVE_BACKWARD".
Resume using the following instruction and the objects in the provided image.

---
The instruction is as follows:
---
{"instruction": '[INSTRUCTION]}
---
{"objects" = [OBJECTS]}
{"actions" = [ACTIONS]}
---
The dictonary that you return should be formatted as python dictonary. Follow these rules:
1. Never leave ',' at the end of the list.
2. All keys of the dictionary should be double-quoted.
3. Insert ``` at the beginning and the end of the dictionary to separate it from the rest of your response.
4. If the statement does not directed toward the robot or is not a request for the robot to perform an action the list should be empty.
"""        
        
        instruction = deepcopy(prompt)
        if instruction.find("[ACTIONS]") != -1:
            instruction = instruction.replace("[ACTIONS]", self.actions)
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
        '''
        results = self.client.chat.completions.create(model=model, messages=self.messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        response = []
        for chunk in results:
            response.append(chunk.choices[0].delta.content)

        ans = [m for m in response if m is not None]
        answer = ''.join([m for m in ans])
        '''
        answer = self.woz(self.messages)
        
        '''
        self.messages.append(
            {
                "role":"assistant", 
                "content": [
                    {"type":"text", "text" : answer},
                ]
            }
        )
        '''
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
