from geometry_msgs.msg import PointStamped, Point
from segment_anything import SamPredictor, sam_model_registry
import torch
import torchvision
import json

class GetMask:
    def __init__(self, hostname, port):
        sever_address = hostname
        sever_port  = port
    
        print("PyTorch version:", torch.__version__)
        print("Torchvision version:", torchvision.__version__)
        print("CUDA is available:", torch.cuda.is_available())

        self.sam = sam_model_registry["default"](checkpoint=model_path)
        if torch.cuda.is_available():
            self.sam.to(device="cuda")

        self.predictor = SamPredictor(sam)
        print('sam model loaded')
    
        print(f"Connecting to {sever_address}:{sever_port}")
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://"+sever_address+":"+sever_port)
        print(f"Connected to {sever_address}:{sever_port}")


    def run(self):
        while True:
            msg = self.socket.recv_json()
            masks = self.recv_msg(msg)
            print('recv_msg')
            self.socke.send(masks)

    def recv_msg(self, msg):
        data = json.dump(msg)
        target_x = data["input_point"][0]
        target_y = data["input_point"][1]
        
        img = data["image"]
        
        input_point = np.array([[target_x, target_y]])
        input_label = np.array([1])

        print('model start')
        self.predictor.set_image(img)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        #print(scores.shape)
        #print(logits.shape)
        print('model end')

        return masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", default="iral-pinky.cs.umbc.edu", required=False,
                        help="hostname for ROS system running transcribe_server.py")
    parser.add_argument("--port", default="8888", required=False,
                        help="port transcribe_server.py is listening on.")
    parser.add_argument("--model_path", default="/home/phiggin1/segment-anything/models/sam_vit_h_4b8939.pth", required=False,
                        help="port transcribe_server.py is listening on.")
    args = parser.parse_args()

    hostname = args.hostname
    port = args.port
    model_path = args.model_path

    mask = GetMask(hostname=hostname,port=port)
    mask.run()