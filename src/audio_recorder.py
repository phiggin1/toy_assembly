#!/usr/bin/python3

import rospy
import pyaudio
import json
import numpy as np
from std_msgs.msg import String

CHANNELS = 1 
CHUNK = 1024
FORMAT = pyaudio.paFloat32
RATE = 16000

rospy.init_node("audio_recorder")

pub = rospy.Publisher("/test", String)

pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

while not rospy.is_shutdown():
    #read in audio clip
    audio = stream.read(CHUNK)
    #get the float array
    float_array = np.fromstring(audio, dtype=float)
    #convert to json string to match RIVR
    float_string = json.dumps(float_array.tolist())
    #publlish
    pub.publish(float_string)

stream.close()
pa.terminate()