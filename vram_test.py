
print("import torch torchvision")
import torch
import torchvision

print("import clip")
import clip
print("import whisper")
import whisper
print("import segment_anything")
from segment_anything import SamPredictor, sam_model_registry

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

clip_model_path =  "/nfs/ada/cmat/users/phiggin1/clip_models/ViT-B-32.pt"
sam_model_path = "/nfs/ada/cmat/users/phiggin1/sam_models/sam_vit_h_4b8939.pth"
whisper_model_path = "/nfs/ada/cmat/users/phiggin1/whisper_models"


print("clip_model_path:", clip_model_path)
print("sam_model_path:", sam_model_path)
print("whisper_model_path:", whisper_model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device:",device)

model, preprocess = clip.load("ViT-B/32", device=device)
print("loaded clip")

whisper_model = whisper.load_model("small", download_root=whisper_model_path)
print("loaded whisper")

sam = sam_model_registry["default"](checkpoint=sam_model_path)
sam.to(device="cuda")
print("loaded sam")

print("loaded all models")