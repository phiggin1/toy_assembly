#!/usr/bin/env python3

import rospy
import pyaudio
import json
import numpy as np
from std_msgs.msg import String

CHUNK = 1024
CHANNELS = 1 
FORMAT = pyaudio.paFloat32
RATE = 16000

rospy.init_node("audio_recorder")

pub = rospy.Publisher("/test", String)

pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

while not rospy.is_shutdown():
    a = stream.read(CHUNK)
    float_array = np.fromstring(a, dtype=float)
    float_string = json.dumps(float_array.tolist())
    #publish string
    print(float_array[0])
    pub.publish(float_string)

stream.close()
pa.terminate()