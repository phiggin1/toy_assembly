#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from openai import OpenAI
import time
from toy_assembly.srv import LLMText, LLMTextRequest, LLMTextResponse
from toy_assembly.srv import LLMImage, LLMImageRequest, LLMImageResponse

class GPTServ:
    def __init__(self):
        self.stop = False
        rospy.init_node('gpt_service_node')
        
        
        key_filename = rospy.get_param("~key_file", "/home/phiggin1/ai.key")
        with open(key_filename, "rb") as key_file:
            key = key_file.read().decode("utf-8")
            
        self.client = OpenAI(
            api_key = key,
        )

        self.llm_img_serv = rospy.Service("/llm_image", LLMImage, self.LLMImage)
        rospy.spin()

    def get_prompt_llm(self, statement):
        objects = ['<red_horse_front_legs>', '<yellow_horse_back_legs>', '<horse_body_blue>', '<horse_body_red>', '<horse_body_yellow>']
        with open("/home/phiggin1/catkin_ws/src/toy_assembly/src/prompt/system.txt") as f:
            system = f.read()
        with open("/home/phiggin1/catkin_ws/src/toy_assembly/src/prompt/prompt.txt") as f:
            prompt = f.read()
        with open("/home/phiggin1/catkin_ws/src/toy_assembly/src/prompt/query.txt") as f:
            query = f.read()

        if query.find("[OBJECTS]") != -1:
            query = query.replace("[OBJECTS]", ", ".join(objects))

        if query.find("[STATEMENT]") != -1:
            query = query.replace("[STATEMENT]", statement)

        messages = []
        messages.append({"role": "system", 
                       "content": system})
        messages.append({"role": "assistant", 
                       "content" : 'Understood. Waiting for next input.'})
        
        messages.append({"role": "user", 
                       "content": prompt})
        messages.append({"role": "assistant", 
                       "content" : 'Understood. Waiting for next input.'})
        
        messages.append({"role": "user", 
                       "content" : query})

        return messages

    def chat_complete(self, messages):
        
        start_time = time.time_ns()
        print(f"{start_time/10**9} :\tSending query")
        response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=messages,
                temperature=0.0,
                max_tokens=8000,
                top_p=0.5,
                frequency_penalty=0.0,
                presence_penalty=0.0)
        end_time = time.time_ns()
        print(f"{end_time/10**9} :\tRespone recieved")
        print(f"latency: {(end_time-start_time)/(10**9)}")
        
        text = response.choices[0].message.content
        
        print(text)

        return text

    def encode_image(self, image):

        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    def LLMImage(self, req):
        statement = req.text
        state = req.state
        encoded_image = encode_image(req.image)

        messages = self.get_prompt_llm(statement, encoded_image, state)

        text = self.chat_complete(messages)

        resp = LLMTextResponse()
        resp.text = text

        return resp



if __name__ == '__main__':
    gpt = GPTServ()

