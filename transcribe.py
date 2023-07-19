
import zmq
import sys
import time
#import whisper


class Transcribe:
    def __init__(self):
        sever_address = "iral-pinky.cs.umbc.edu"
        sever_port  = "8888"


        #self.whisper_model = whisper.load_model("base")   

        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://"+sever_address+":"+sever_port)
        self.tmp_audio_filename = '/tmp/audio.mp3'

    def run(self):
        while True:
            msg = self.socket.recv()
            self.recv_msg(msg)
        

    def recv_msg(self, msg):
        #save to /tmp/audio.mp3
        '''
        binary_file = open(self.tmp_audio_filename, "wb")
        binary_file.write(msg)
        binary_file.close()

        #get transcription from whisper
        result = self.whisper_model.transcribe(self.tmp_audio_filename) 

        print(result["text"])
        self.socket.send(result["text"])
        '''

        self.socket.send_string("transcript")


if __name__ == '__main__':
    t = Transcribe()
    t.run()

