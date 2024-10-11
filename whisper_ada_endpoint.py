
import torch
import torchvision
#import clip
#import whisper
from faster_whisper import WhisperModel
from segment_anything import SamPredictor, sam_model_registry
from scipy.io.wavfile import write as wavfile_writer
import numpy as np
import zmq
import argparse
import PIL
import json
import time

class AdaEndPoint:
    def __init__(self, hostname, port, whisper_model_path):
        print(f"PyTorch version: {torch.__version__}")
        print(f"Torchvision version: {torchvision.__version__}")
        print(f"CUDA is available: {torch.cuda.is_available()}, version: {torch.version.cuda}")
        print(torch.backends.cudnn.version())

        print("whisper_model_path:", whisper_model_path)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("device:",self.device)

        sever_address = hostname
        server_port  = port
        
        print(f"{time.time_ns()}: loading whipser")
        #self.whisper_model = whisper.load_model("large", download_root="/nfs/ada/cmat/users/phiggin1/whisper_models")  
        self.whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        
        self.get_mem_usage(self.device)
        
        print(f"Connecting to {sever_address}:{server_port}")
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://"+sever_address+":"+server_port)
        print(f"Connected to {sever_address}:{server_port}")

        self.tmp_audio_filename = '/tmp/audio.mp3'

    def get_mem_usage(self, device):
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(device)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(device)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(device)/1024/1024/1024))

    def run(self):
        while True:
            msg = self.socket.recv_json()

            msg_type = msg["type"]
            print(f"{time.time_ns()}: Message recieved type: {msg_type}")
            start_time = time.time()
            
            if msg_type == "whisper":
                resp = self.process_whisper(msg)
            else:
                resp = {}

            self.socket.send_json(resp)
            end_time = time.time()
            print(f"{time.time_ns()}: Message replied type: {msg_type}, took {end_time-start_time} second")
            
    
    def process_whisper(self, data):
        audio_data = data["data"]
        sample_rate =  data["sample_rate"]
        context = data["context"]
        audio = np.fromstring(audio_data[1:-1], dtype=float, sep=',')
        wavfile_writer(self.tmp_audio_filename, sample_rate, audio)

        #context = "move forward backward up down left right turn tilt rotate pickup grab open close"
        #print(context)

        #get transcription from whisper
        '''
        result = self.whisper_model.transcribe(self.tmp_audio_filename, initial_prompt=context) 
        text = result["text"]
        '''
        segments, info = self.whisper_model.transcribe(self.tmp_audio_filename, beam_size=5, word_timestamps=True, initial_prompt=context, language="en")
        
        text = "" 
        for s in segments:
          text+=s.text
          print(f"{s.start} to {s.end}: {s.text}")
        
        print(text)
        response = {"type":"whisper",
                    "text":text
        }
        
        return response
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", default="iral-pinky.cs.umbc.edu", required=False,
                        help="hostname for ROS system running ada_client.py")
    parser.add_argument("--port", default="8888", required=False,
                        help="port transcribe_server.py is listening on.")
    parser.add_argument("--sam_model_path", default="/nfs/ada/cmat/users/phiggin1/sam_models/sam_vit_h_4b8939.pth", required=False,
                        help="Path to the SAM model")
    parser.add_argument("--whisper_model_path", default="/nfs/ada/cmat/users/phiggin1/whisper_models", required=False,
                        help="Path to the Whisper model")
    parser.add_argument("--clip_model_path", default="/nfs/ada/cmat/users/phiggin1/clip_models/ViT-B-32.pt", required=False,
                        help="Path to the CLIP model")
    parser.add_argument("--torch_home_path", default="/nfs/ada/cmat/users/phiggin1/torch_home", required=False,
                        help="Path to the Torch Home directory.")
    
    args = parser.parse_args()

    hostname = args.hostname
    port = args.port
    whisper_model_path = args.whisper_model_path

    endpoint = AdaEndPoint(hostname=hostname, port=port, whisper_model_path=whisper_model_path)
    endpoint.run()
