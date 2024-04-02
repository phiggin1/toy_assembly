#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from openai import OpenAI
import time
from toy_assembly.srv import LLMText, LLMTextRequest, LLMTextResponse

class GPTServ:
    def __init__(self):
        self.stop = False
        rospy.init_node('gpt_service_node')
        
        '''
        key_filename = "/home/phiggin1/ai.key"
        with open(key_filename, "rb") as key_file:
            key = key_file.read().decode("utf-8")


        self.client = OpenAI(
            api_key = key,
        )
        '''
        #self.llm_img_serv = rospy.Service("/llm_image", LLMImage, self.LLMImage)
        self.llm_text_serv = rospy.Service("/llm_text", LLMText, self.LLMText)
        rospy.spin()

    def get_prompt(self, statement):
        objects = ['<red_horse_front_legs>', '<yellow_horse_back_legs>', '<horse_body_blue>', '<horse_body_red>', '<horse_body_yellow>']

        prompt_mesgage = f"""
You are a robotic system that is working togheter with a human to build a slot togeter toy horse.

You can see 5 different 'objects': {objects}.

The human will tell you what part they will pick up and what part you (the robot should) pick up.

When told by the human what parts are to be picked up you sould respond with a python ndictionary.
The dictionary should have two keys.

dictonary["robot"] : the object that the robot should pickup
dictonary["human"] : the object that the human will pickup
                
Please do not begin working until I say "Start working." Instead, simply output the message "Waiting for next input." Understood?        
"""
        
        query = "Start working.\n"
        prompt = []
        prompt.append({"role": "system", 
                       "content": prompt_mesgage})
        prompt.append({"role": "assistant", 
                       "content" : 'Understood. Waiting for next input.'})
        prompt.append({"role": "user", 
                       "content" : query+statement})

        return prompt

    def chat_complete(self, statement):
        start_time = time.time_ns()
        print(f"{start_time/10**9} :\tSending query")

        prompt = self.get_prompt(statement)
        
        '''
        response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=prompt,
                temperature=0.1,
                max_tokens=8000,
                top_p=0.5,
                frequency_penalty=0.0,
                presence_penalty=0.0)
        
        end_time = time.time_ns()
        print(f"{end_time/10**9} :\tRespone recieved")
        print(f"latency: {(end_time-start_time)/(10**9)}")

        text = response.choices[0].message.content
        '''

        text = """Sure! Here is the dictionary with the objects we should pick up:

{
  "robot": "<horse_body_blue>",
  "human": "<red_horse_front_legs>"
}"""

        return text
        '''
        a = text.find('{')
        b = text.find('}')+1
        text_json = text[a:b]
        json_dict = json.loads(text_json)

        return json_dict
        '''

    def LLMText(self, req):
        statement = req.text

        text = self.chat_complete(statement)

        resp = LLMTextResponse()
        resp.text = text

        return resp



if __name__ == '__main__':
    gpt = GPTServ()

