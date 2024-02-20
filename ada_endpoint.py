
import torch
import torchvision
import clip
import whisper
from segment_anything import SamPredictor, sam_model_registry

from scipy.io.wavfile import write as wavfile_writer
import numpy as np
import json
import zmq
import argparse
import PIL

class AdaEndPoint:
    def __init__(self, hostname, port, sam_model_path, whisper_model_path, clip_model_path):
        print("PyTorch version:", torch.__version__)
        print("Torchvision version:", torchvision.__version__)
        print("CUDA is available:", torch.cuda.is_available())

        print("sam_model_path:", sam_model_path)
        print("whisper_model_path:", whisper_model_path)
        print("clip_model_path:", clip_model_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:",self.device)

        sever_address = hostname
        server_port  = port

        print("loading whipser")
        self.whisper_model = whisper.load_model("medium", download_root="/nfs/ada/cmat/users/phiggin1/whisper_models")  
        
        print("loading sam")
        self.sam = sam_model_registry["default"](checkpoint=sam_model_path)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)

        print("loading clip")
        self.clip_model, self.clip_preprocess = clip.load(name=clip_model_path, device=self.device)
                
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

        print(f"Connecting to {sever_address}:{server_port}")
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://"+sever_address+":"+server_port)
        print(f"Connected to {sever_address}:{server_port}")

        self.sample_rate = 16000
        self.tmp_audio_filename = '/tmp/audio.mp3'

    def run(self):
        while True:
            msg = self.socket.recv_json()
            print('recieved')

            msg_type = msg["type"]
            if msg_type == "sam":
                resp = self.process_sam(msg)
            elif msg_type == "clip":
                resp = self.process_clip(msg)
            elif msg_type == "whisper":
                resp = self.process_whisper(msg)
            else:
                resp = {}

            self.socket.send_json(resp)
            print('replied')

    def process_sam(self, data):
        target_x = data["target_x"]
        target_y = data["target_y"]
        
        img = np.asarray(data["image"], dtype=np.uint8)

        
        input_point = np.array([[target_x, target_y]])
        input_label = np.array([1])

        print('sam start')
        print(img.shape)

        self.predictor.set_image(img)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        print('sam end')

        print(masks.shape)

        response = {"type":"sam",
                    "masks":masks.tolist(),
                    "scores":scores.tolist()
        }

        return response
    
    def process_whisper(self, data):
        data = data["data"]
        audio = np.fromstring(data[1:-1], dtype=float, sep=',')
        wavfile_writer(self.tmp_audio_filename, self.sample_rate, audio)

        #get transcription from whisper

        print('whisper start')
        result = self.whisper_model.transcribe(self.tmp_audio_filename) 
        print('whisper end')

        print(result["text"])
        response = {"type":"whisper",
                    "text":result["text"]

        }
        return response
    
    def process_clip(self, data):

        print('clip model start')
        #images = data["images"]
        raw_images = []
        for img in data["images"]:
            #images.append(preprocess(image))
            raw_images.append( self.clip_preprocess(PIL.Image.fromarray(np.asarray(img, dtype=np.uint8))).unsqueeze(0) )
        raw_text = []
        for t in data["text"]:
            raw_text.append(t)
    
        images = torch.cat(raw_images, 0).to(self.device)
        text = clip.tokenize(raw_text).to(self.device)

        #image_features = self.clip_model.encode_image(images)
        #text_features = self.clip_model.encode_text(text)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.clip_model(images, text)
            probs = logits_per_text.softmax(dim=-1).cpu().numpy()
        print('clip model end')

        print("logits:", logits_per_text) 
        print("Label probs:", probs)  
    
        response = {"type":"clip",
                    "probs":probs.tolist()
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
    args = parser.parse_args()

    hostname = args.hostname
    port = args.port
    sam_model_path = args.sam_model_path
    whisper_model_path = args.whisper_model_path
    clip_model_path = args.clip_model_path

    endpoint = AdaEndPoint(hostname=hostname, port=port, sam_model_path=sam_model_path, whisper_model_path=whisper_model_path, clip_model_path=clip_model_path)
    endpoint.run()