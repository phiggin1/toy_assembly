
import torch
import torchvision
import clip
import whisper
from segment_anything import SamPredictor, sam_model_registry

import numpy as np
import json
import zmq
import argparse

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

        self.whisper_model = whisper.load_model("small", download_root="/nfs/ada/cmat/users/phiggin1/whisper_models")  
        
        self.sam = sam_model_registry["default"](checkpoint=sam_model_path)
        if torch.cuda.is_available():
            self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)

        self.clip_model, self.clip_preprocess = clip.load(name=clip_model_path, device=self.device)

        print(f"Connecting to {sever_address}:{server_port}")
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://"+sever_address+":"+server_port)
        print(f"Connected to {sever_address}:{server_port}")

        self.tmp_audio_filename = '/tmp/audio.mp3'

    def run(self):
        while True:
            msg = self.socket.recv_json()
            print('recv_msg')

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

    def process_sam(self, data):
        target_x = data["target_x"]
        target_y = data["target_x"]
        
        img = np.asarray(data["image"], dtype=np.uint8)
        
        input_point = np.array([[target_x, target_y]])
        input_label = np.array([1])

        print('sam start')
        self.predictor.set_image(img)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        print('sam end')

        response = {"type":"sam",
                    "masks":masks.tolist(),
                    "scores":scores.tolist()

        }


        return response
    
    def process_whisper(self, data):
        print('recv_msg')

        #get bytes from json
        audio_bytes = data["data"]

        #save to /tmp/audio.mp3
        binary_file = open(self.tmp_audio_filename, "wb")
        binary_file.write(audio_bytes)
        binary_file.close()

        #get transcription from whisper
        result = self.whisper_model.transcribe(self.tmp_audio_filename) 

        print(result["text"])
        response = {"type":"whisper",
                    "text":result["text"]

        }
        return response
    
    def process_clip(self, data):

        print('clip model start')
        #images = data["images"]
        images = []
        for img in data["images"]:
            images.append(np.asarray(img, dtype=np.uint8))
        text = []
        for t in data["text"]:
            text.append(t)
    
        print(text)

        #org clip_image = self.clip_preprocess(images).unsqueeze(0).to(self.device)
        clip_image = self.clip_preprocess(images).to(self.device)
        clip_text = clip.tokenize(text).to(self.device)
    
        with torch.no_grad():
            #image_features = self.clip_model.encode_image(clip_image)
            #text_features = self.clip_model.encode_text(text)
            logits_per_image, logits_per_text = self.clip_model(clip_image, clip_text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        print("Label probs:", probs)  
        print('clip model end')
    
        response = {"type":"clip",
                    "probs":probs
        }

        return response
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", default="iral-pinky.cs.umbc.edu", required=False,
                        help="hostname for ROS system running transcribe_server.py")
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

    mask = AdaEndPoint(hostname=hostname, port=port, sam_model_path=sam_model_path, whisper_model_path=whisper_model_path, clip_model_path=clip_model_path)
    mask.run()