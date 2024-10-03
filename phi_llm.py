
import torch
import torchvision
import numpy as np
import zmq
import argparse
import json
import time
from PIL import Image
from vllm import LLM, SamplingParams



class AdaEndPoint:
    def __init__(self, hostname, port):
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"Torchvision version: {torchvision.__version__}")
        print(f"CUDA is available: {torch.cuda.is_available()}, version: {torch.version.cuda}")
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("device:",self.device)

        sever_address = hostname
        server_port  = port

        
        print(f"{time.time_ns()}: start model load")
        start_time = time.time()
        
        #LLAMA3/PHI3
        #model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        '''
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        self.llm = LLM(
          model=model_name, 
          revision="65be4e00a56c16d036e9cbe96b0b35f8aa0f84b0",
          dtype="float16"
        )
        '''
        #model_path = ".cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/65be4e00a56c16d036e9cbe96b0b35f8aa0f84b0/"
        model_path = "/home/phiggin1/cmat_ada/users/phiggin1/phi3_model"
        self.llm = LLM(
          model=model_path, 
          dtype="float16"
        )
        
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(temperature=0,max_tokens=64)

        
        print(f"{time.time_ns()}: finish model load")
        end_time = time.time()
        
        print(f"model load took {end_time-start_time} seconds")        
        
                
        self.get_mem_usage(self.device)

        print(f"Connecting to {sever_address}:{server_port}")
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://"+sever_address+":"+server_port)
        print(f"Connected to {sever_address}:{server_port}")
        
        
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

          ["MOVE_TO", "Move the are to a given location"],
          
          ["OTHER", "Any other possible command"]
        ]
        self.actions = ""
        for a in actions_list:
          self.actions += " - "+a[0]+" : " + a[1]+"\n"

        
        self.system = """
You are an excellent interpreter of human instructions for basic tasks. You are working with a human to jointly perform a simple collaborative task. In this task you are a robot working with a human to build a slot together toy.

'actions' = {[ACTIONS]}

For a given statement determine if the statement is directed to the robot, is not a request or is not action.
If it is not return the action the robot should as a python dictonary.
The value MUST be one of the actions in the 'actions' list or empty no other values should be passed.
The dictionary has one key.
---
- dictonary["action"] : A list of the actions that the robot should take, taken from the actions list above.
---
-------------------------------------------------------
"""
        
        if self.system.find("[ACTIONS]") != -1:
            self.system = self.system.replace("[ACTIONS]", self.actions)
        
        self.chat = [
            {'role': 'system', 'content': self.system},
        ] 

    def get_mem_usage(self, device):
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(device)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(device)/1024/1024/1024))

    def run(self):
       
        while True:
            msg = self.socket.recv_json()

            #text = input("Text:")
            #msg = {
            #     "text":text,
            #     "type":"llm"
            #}

            msg_type = msg["type"]
        
            start_time = time.time()
            print(f"{time.time_ns()}: Message recieved type: {msg_type} \n {msg}")

            if msg_type == "llm":
              resp = self.process_llm(msg)
            else:
                resp = {}

            print(resp)

            self.socket.send_json(resp)
            end_time = time.time()
            print(f"{time.time_ns()}: Message replied type: {msg_type}, took {end_time-start_time} seconds")

            

    def process_llm(self, data):
        text = data["text"]
        
        #TODO better filtering
        if len(text) < 3:
          response = {
            "type":"llm",
            "text":"",
            "error":"invalid transcript"
          }
          return response

        prompt = """
Start working using the instruction given below.
---
The instruction is as follows:
---
{"instruction": "[STATEMENT]"}
---
The dictionary that you return should be formatted as python dictionary. Follow these rules:
1. Never leave ',' at the end of the list.
2. All keys of the dictionary should be double-quoted.
3. All values of the dictionary should be double-quoted.
4. Insert ``` at the beginning and the end of the dictionary to separate it from the rest of your response.
5. Only use the actions listed in the list of actions.
6. If the statement does not directed toward the robot or is not a request for the robot to perform an action the list should be empty.
7. If the statement requires outside context such as refering to an object the list should be empty.
7. If the statement referst to any object or location the action list should be empty.
"""  
        if prompt.find("[STATEMENT]") != -1:
            prompt = prompt.replace("[STATEMENT]", text)
        
        '''
        self.chat = [
            {'role': 'system', 'content': self.system},
            {'role': 'user', 'content': prompt}
        ] 
        '''
        self.chat.append({'role': 'user', 'content': prompt})

        conversations = self.tokenizer.apply_chat_template(self.chat, tokenize=False)

        #print(type(conversations))
        #print(len(conversations))
        #print(conversations)
        tokens = self.tokenizer.tokenize(conversations)
        num_tokens = len(tokens)
        print(num_tokens)
         
        while num_tokens > 4096:
          del self.chat[1:2]
          conversations = self.tokenizer.apply_chat_template(self.chat, tokenize=False)
          tokens = self.tokenizer.tokenize(conversations)
          num_tokens = len(tokens)
          print(num_tokens)
        
        print(f"{time.time_ns()}: starting inference")
        start_time = time.time()

        outputs = self.llm.generate(conversations, sampling_params=self.sampling_params)
        
        print(f"{time.time_ns()}: finished inference") 
        end_time = time.time()
        print(f"inference took {end_time-start_time} seconds")
        print("##################################")
        
        text = []
        for o in outputs:
          generated_text = o.outputs[0].text
          text.append(generated_text)
          
        text = ''.join(text)
        print(text)
        
        self.chat.append({'role': 'assistant', 'content': text})
        
        print("##################################")

        self.get_mem_usage(self.device)

        response = {"type":"llm",
                    "text":text
        }
        
        return response

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", default="iral-pinky.cs.umbc.edu", required=False,
                        help="hostname for ROS system running ada_client.py")
    parser.add_argument("--port", default="8877", required=False,
                        help="port transcribe_server.py is listening on.")


    
    args = parser.parse_args()

    hostname = args.hostname
    port = args.port

    endpoint = AdaEndPoint(hostname=hostname, port=port)
    endpoint.run()
