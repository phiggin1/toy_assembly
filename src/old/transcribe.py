
import zmq
import sys
import time
import whisper
import argparse

class Transcribe:
    def __init__(self, hostname, port):
        sever_address = hostname
        sever_port  = port

        self.whisper_model = whisper.load_model("small", download_root="/nfs/ada/cmat/users/phiggin1/whisper_models")  
        
        print(f"Connecting to {sever_address}:{sever_port}")
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://"+sever_address+":"+sever_port)
        print(f"Connected to {sever_address}:{sever_port}")

        self.tmp_audio_filename = '/tmp/audio.mp3'

    def run(self):
        while True:
            msg = self.socket.recv()
            text = self.recv_msg(msg)
            self.socket.send_string(text)

    def recv_msg(self, msg):
        print('recv_msg')

        #save to /tmp/audio.mp3
        binary_file = open(self.tmp_audio_filename, "wb")
        binary_file.write(msg)
        binary_file.close()

        #get transcription from whisper
        result = self.whisper_model.transcribe(self.tmp_audio_filename) 

        print(result["text"])

        return result["text"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", default="iral-pinky.cs.umbc.edu", required=False,
                        help="hostname for ROS system running transcribe_server.py")
    parser.add_argument("--port", default="8888", required=False,
                        help="port transcribe_server.py is listening on.")
    args = parser.parse_args()
    hostname = args.hostname
    port = args.port
    
    t = Transcribe(hostname=hostname,port=port)
    t.run()

