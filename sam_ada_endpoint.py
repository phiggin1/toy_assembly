import os
import cv2
import json
import torch
import numpy as np
import torchvision
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model,  predict
import groundingdino.datasets.transforms as T
from PIL import Image
import zmq
import time
import argparse

"""
Hyper parameters
"""
TEXT_PROMPT = "car. tire."
IMG_PATH = "notebooks/images/truck.jpg"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("/home/phiggin1/cmat_ada/users/phiggin1/images/")
DUMP_JSON_RESULTS = True
NMS_THRESHOLD = 0.8

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def load_image(cv_img):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_source = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)

    return image, image_transformed

# environment settings
# use bfloat16
class SamEndPoint:
    def __init__(self, hostname, port, sam2_checkpoint, sam2_model_config, grounding_dino_checkpoint, grounding_dino_config):
        sever_address = hostname
        server_port  = port

        # build SAM2 image predictor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sam2_model = build_sam2(
            config_file=sam2_model_config, 
            ckpt_path=sam2_checkpoint, 
            device=self.device
        )
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=grounding_dino_config, 
            model_checkpoint_path=grounding_dino_checkpoint,
            device=self.device
        )
        self.get_mem_usage(self.device)
        
        print(f"Connecting to {sever_address}:{server_port}")
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://"+sever_address+":"+server_port)
        print(f"Connected to {sever_address}:{server_port}")

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
            
            if msg_type == "sam":
                resp = self.process_sam(msg)
            else:
                resp = {}

            self.socket.send_json(resp)
            end_time = time.time()
            print(f"{time.time_ns()}: Message replied type: {msg_type}, took {end_time-start_time} second")
            
    def process_sam(self, msg):
        cv_img = np.asarray(msg["image"], dtype=np.uint8)        
        text = msg["text"]
        
        print(f"text prompt: {text}")
        # setup the input image and text prompt for SAM 2 and Grounding DINO
        # VERY important: text queries need to be lowercased + end with a dot

        image_source, image = load_image(cv_img)

        self.sam2_predictor.set_image(image_source)

        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image,
            caption=text,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            remove_combined=True
        )

        # NMS post process
        print(f"Before NMS: {len(boxes)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.cat([boxes[:,2:], boxes[:,:2]], dim=1),
            confidences, 
            NMS_THRESHOLD
        ).numpy().tolist()
        boxes = boxes[nms_idx]
        confidences = confidences[nms_idx]
        labels = labels[nms_idx]
        print(f"After NMS: {len(detections.xyxy)} boxes")


        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # FIXME: figure how does this influence the G-DINO model
        torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = confidences.numpy().tolist()
        class_names = labels
        class_ids = np.array(list(range(len(class_names))))
        
        #Label is just class name
        labels = [
            f"{class_name}"
            for class_name, class_id 
            in zip(class_names, class_ids)
        ]        
        '''
        #Label is a unique numericla ID
        labels = [
            f"object_{class_id}"
            for class_id 
            in zip(class_ids)
        ]
        '''
        """
        Visualize image with supervision useful API
        """
        img = cv_img.copy()
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )
        box_annotator = sv.BoxCornerAnnotator(thickness=1, corner_length=10)
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator(text_scale=0.25, text_thickness=1, text_padding=2)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        """
        Dump the results in standard format and save as json files
        """
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        input_boxes = input_boxes.tolist()
        scores = scores.tolist()
        # save the results in standard format
        results = {
            "type":"sam",
            "annotated_image":annotated_frame.tolist(),
            "annotations" : [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }

        return results
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", default="iral-pinky.cs.umbc.edu", required=False,
                        help="hostname for ROS system running ada_client.py")
    parser.add_argument("--port", default="8899", required=False,
                        help="port transcribe_server.py is listening on.")
    parser.add_argument("--sam2_checkpoint", default="/home/phiggin1/cmat_ada/users/phiggin1/grouned_sam2_models/checkpoints/sam2.1_hiera_large.pt", required=False,
                        help="port transcribe_server.py is listening on.")
    parser.add_argument("--sam2_model_config", default="configs/sam2.1/sam2.1_hiera_l.yaml", required=False,
                        help="port transcribe_server.py is listening on.")
    parser.add_argument("--grounding_dino_checkpoint", default="/home/phiggin1/cmat_ada/users/phiggin1/grouned_sam2_models/gdino_checkpoints/groundingdino_swint_ogc.pth", required=False,
                        help="port transcribe_server.py is listening on.")
    parser.add_argument("--grounding_dino_config", default="/home/phiggin1/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py", required=False,
                        help="port transcribe_server.py is listening on.")
    parser.add_argument("--box_threshold", default=0.35, required=False,
                        help="port transcribe_server.py is listening on.")
    parser.add_argument("--text_threshold", default=0.25, required=False,
                        help="port transcribe_server.py is listening on.")

    args = parser.parse_args()
    hostname = args.hostname
    port = args.port
    sam2_checkpoint = args.sam2_checkpoint
    sam2_model_config = args.sam2_model_config
    grounding_dino_checkpoint = args.grounding_dino_checkpoint
    grounding_dino_config = args.grounding_dino_config
    
    endpoint = SamEndPoint(hostname=hostname, port=port, sam2_checkpoint=sam2_checkpoint, sam2_model_config=sam2_model_config, grounding_dino_checkpoint=grounding_dino_checkpoint, grounding_dino_config=grounding_dino_config)
    endpoint.run()
