import argparse
import os
import time
import random

from PIL import Image

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

import zmq
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", default="iral-pinky.cs.umbc.edu", required=False,
                        help="hostname for ROS system running ada_client.py")
    parser.add_argument("--port", default="8877", required=False,
                        help="port transcribe_server.py is listening on.")
    parser.add_argument("--cfg-path", default="minigpt4_eval_configs/minigpt4_eval.yaml", 
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args

class LLMEndPoint:
    def __init__(self, args):
        print("PyTorch version:", torch.__version__)
        print("Torchvision version:", torchvision.__version__)
        print("CUDA is available:", torch.cuda.is_available())

        sever_address = args.hostname
        server_port  = args.port
        
        print(f"{time.time_ns()} : \t Initializing Chat")
        self.cfg = Config(args)

        self.num_beams = 1
        self.temperature = 1.0
        
        self.model_config = self.cfg.model_cfg
        self.model_config.device_8bit = args.gpu_id
        self.model_cls = registry.get_model_class(self.model_config.arch)
        self.model = self.model_cls.from_config(self.model_config).to('cuda:{}'.format(args.gpu_id))

        self.conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                    'pretrain_llama2': CONV_VISION_LLama2}
        self.CONV_VISION = self.conv_dict[self.model_config.model_type]

        self.vis_processor_cfg = self.cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(self.vis_processor_cfg.name).from_config(self.vis_processor_cfg)

        self.stop_words_ids = [[835], [2277, 29937]]
        self.stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in self.stop_words_ids]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=self.stop_words_ids)])

        self.chat = Chat(self.model, self.vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=self.stopping_criteria)
        print(f"{time.time_ns()} : \t Initialization Finished")
        
        print(f"torch.cuda.memory_allocated: {(torch.cuda.memory_allocated(0)/1024/1024/1024)}GB")
        print(f"torch.cuda.memory_reserved: {(torch.cuda.memory_reserved(0)/1024/1024/1024)}GB")
        print(f"torch.cuda.max_memory_reserved: {(torch.cuda.max_memory_reserved(0)/1024/1024/1024)}GB")


        self.chat_state = self.CONV_VISION.copy()
        self.img_list = []

        print(f"Connecting to {sever_address}:{server_port}")
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://"+sever_address+":"+server_port)
        print(f"Connected to {sever_address}:{server_port}")

    def run(self):
        while True:
            msg = self.socket.recv_json()
            msg_type = msg["type"]
            print(f"{time.time_ns()}: Message recieved type: {msg_type}")
            if msg_type == "llm_image":
                resp = self.process_img(msg)
            elif msg_type == "llm_ask":
                resp = self.process_text(msg)
            else:
                resp = {}

            self.socket.send_json(resp)
            print(f"{time.time_ns()}: Message replied type: {msg_type}")


    def process_img(self, msg):

        img = np.asarray(msg["image"], dtype=np.uint8)
        img = Image.fromarray(img)

        llm_message = self.chat.upload_img(img, self.chat_state, self.img_list)
        self.chat.encode_img(self.img_list)

        response = {"type":"llm_ask"
        }

        return response
    
    def ask_llm(user_message, img_list, chat_state):
        print(f"{time.time_ns()} | \t ask start")
        self.chat.ask(user_message, chat_state)
        llm_message = self.chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=self.num_beams,
                                temperature=self.temperature,
                                max_new_tokens=300,
                                max_length=2000)[0]
        print(f"{time.time_ns()} | \t answer end")
        print(f"{time.time_ns()} | \t llm_message: '{llm_message}'")

        response = {"type":"llm_ask",
                    "text":llm_message,
        }

        return response
        
if __name__ == '__main__':
    args = parse_args()
    endpoint = LLMEndPoint(args=args)
    endpoint.run()