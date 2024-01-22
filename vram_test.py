

from PIL import Image

import json
import zmq
import sys
import time
import argparse

import torch
import torchvision

import clip
import whisper
from segment_anything import SamPredictor, sam_model_registry

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

clip_model_path =  "/nfs/ada/cmat/users/phiggin1/clip_models/ViT-B-32.pt"
sam_model_path = "/nfs/ada/cmat/users/phiggin1/sam_models/sam_vit_h_4b8939.pth"
whisper_model_path = "/nfs/ada/cmat/users/phiggin1/whisper_models"

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)


whisper_model = whisper.load_model("small", download_root=whisper_model_path)

sam = sam_model_registry["default"](checkpoint=sam_model_path)
sam.to(device="cuda")

print("loaded all models")