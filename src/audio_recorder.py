#!/usr/bin/python3

import rospy
import pyaudio
import numpy as np
from std_msgs.msg import Float32MultiArray


rospy.init_node("audio_recorder")

publish_rate = 10
ros_pub_rate = rospy.Rate(publish_rate)

CHANNELS = 1 
FORMAT = pyaudio.paFloat32
RATE = 16000

print(RATE/publish_rate)
print(FORMAT)

CHUNK = 1024

pub = rospy.Publisher("/audio", Float32MultiArray, queue_size=10)

pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

while not rospy.is_shutdown():
    #read in audio clip
    audio = stream.read(CHUNK)

    #get the float array
    float_array = np.fromstring(audio, dtype=np.float32)
    
    rospy.loginfo(float_array.shape)

    float_array_msg = Float32MultiArray()
    float_array_msg.data = float_array

    #publish
    pub.publish(float_array_msg)
    #ros_pub_rate.sleep()

stream.close()
pa.terminate()
