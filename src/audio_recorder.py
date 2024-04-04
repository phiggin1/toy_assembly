#!/usr/bin/python3

import rospy
import pyaudio
import numpy as np
from std_msgs.msg import Float32MultiArray

CHANNELS = 1 
CHUNK = 1024
FORMAT = pyaudio.paFloat32
RATE = 16000

rospy.init_node("audio_recorder")

pub = rospy.Publisher("/audio", Float32MultiArray)

pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

while not rospy.is_shutdown():
    #read in audio clip
    audio = stream.read(CHUNK)

    #get the float array
    float_array = np.fromstring(audio, dtype=float)
    float_array_msg = Float32MultiArray()
    float_array_msg.data = float_array

    #publish
    pub.publish(float_array_msg)

stream.close()
pa.terminate()