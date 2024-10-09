
import torch
import torchvision
import numpy as np
from copy import deepcopy
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
        
        fp_actions = "/home/phiggin1/toy_assembly/prompts/actions.txt"
        with open(fp_actions) as f:
            self.actions = f.read()

        fp_system = "/home/phiggin1/toy_assembly/prompts/phi_system.txt"
        with open(fp_system) as f:
            self.system = f.read()
        if self.system.find("[ACTIONS]") != -1:
            self.system = self.system.replace("[ACTIONS]", self.actions)
        print(self.system)

        fp_prompt = "/home/phiggin1/toy_assembly/prompts/phi_prompt.txt"
        with open(fp_prompt) as f:
            self.prompt = f.read()
        if self.prompt.find("[ACTIONS]") != -1:
            self.prompt = self.prompt.replace("[ACTIONS]", self.actions)
        print(self.prompt)

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

        instruction = deepcopy(self.prompt)

        if instruction.find("[STATEMENT]") != -1:
            instruction = instruction.replace("[STATEMENT]", text)
        
        print(f"number of messages in chat:{len(self.chat)}")
        '''
        self.chat = [
            {'role': 'system', 'content': self.system},
            {'role': 'user', 'content': instruction}
        ]
        ''' 
        
        self.chat.append({'role': 'user', 'content': instruction})



        conversations = self.tokenizer.apply_chat_template(self.chat, tokenize=False)

        tokens = self.tokenizer.tokenize(conversations)
        num_tokens = len(tokens)
        print(num_tokens)
        while num_tokens > 3000:
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

        #try and keep the chat log size down
        # phi seems to get odd when it gets close
        # to the contxt limit
        if len(self.chat) > 20:
            del self.chat[1:2]
        
        self.get_mem_usage(self.device)
        
        print("##################################")


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
