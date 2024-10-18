#!/usr/bin/python3

import rospy
import pyaudio
import numpy as np
from rivr_ros.msg import RivrAudio


class AudioStreamer:
    def __init__(self):
        rospy.init_node("audio_recorder")
        rospy.on_shutdown(self.shutdownhook)

        FORMAT = pyaudio.paFloat32
        CHANNELS = 1 
        RATE = 48000
        CHUNK = 1024
        publish_rate = 2

        print(f"rate: {RATE}hz, format: {FORMAT}, publish rate:{publish_rate}hz")

        self.pub = rospy.Publisher("/audio", RivrAudio, queue_size=10)

        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        audio_msg = RivrAudio()
        audio_msg.sample_rate = RATE

        dt = 1/publish_rate
        t = rospy.Time.now()
        next_publish = t

        buffer = np.array([], dtype=np.float32)
        while not rospy.is_shutdown():

            t = rospy.Time.now()

            #rospy.loginfo(f"{next_publish}, {t}")

            #read in audio clip
            audio = self.stream.read(CHUNK)
            #get the float array
            float_array = np.fromstring(audio, dtype=np.float32)
            buffer = np.concatenate((buffer, float_array))
            #print(f"{buffer.shape}, {np.max(buffer)}")
            
            if t > next_publish:
                #publish
                #rospy.loginfo(f"shape: {buffer.shape}, max: {np.max(buffer)}")
                audio_msg.data = buffer
                self.pub.publish(audio_msg)
                next_publish = t + rospy.Duration(dt)
                buffer = np.array([], dtype=np.float32)
                

    def shutdownhook(self):
        self.stream.close()
        self.pa.terminate()

if __name__ == '__main__':
    stream = AudioStreamer()

