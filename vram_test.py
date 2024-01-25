
print("import torch torchvision cv2 PIL")
import torch
import torchvision
#import cv2
from PIL import Image
import numpy as np

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

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print("loaded clip")

whisper_model = whisper.load_model("small", download_root=whisper_model_path)
print("loaded whisper")

sam = sam_model_registry["default"](checkpoint=sam_model_path)
sam.to(device="cuda")
predictor = SamPredictor(sam)
print("loaded sam")

print("loaded all models")

print("whisper model start")
# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("like_this.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

# detect the spoken language
_, probs = whisper_model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(whisper_model, mel, options)

# print the recognized text
print(result.text)
print("whisper model end")


sam_image = Image.open("CLIP.png").convert("RGB")
sam_image = np.array(sam_image)

#cv2.imread('CLIP.png')
#sam_image = #cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)

print('sam model start')
input_point = np.array([[0, 0]])
input_label = np.array([1])
predictor.set_image(sam_image)
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
print(scores.shape)
print(logits.shape)
print('sam model end')

print('clip model start')
clip_image = clip_preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
with torch.no_grad():
    image_features = clip_model.encode_image(clip_image)
    text_features = clip_model.encode_text(text)
    
    print(image_features.shape)

    logits_per_image, logits_per_text = clip_model(clip_image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  
print("[[0.9927937  0.00421068 0.00299572]]")
print('clip model end')