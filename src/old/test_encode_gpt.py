#!/usr/bin/env python3


import io
import time
import base64
from openai import OpenAI
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

class Test:
    def __init__(self):
        rospy.init_node('test', anonymous=True)
        self.bridge = CvBridge()
        
        key_filename = rospy.get_param("~key_file", "/home/phiggin1/ai.key")
        with open(key_filename, "rb") as key_file:
            key = key_file.read().decode("utf-8")
            
        self.client = OpenAI(
            api_key = key,
        )

        rgb_image_topic = f"/unity/camera/right/rgb/image_raw"

        image = rospy.wait_for_message(rgb_image_topic, Image)

        cv_img = self.bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        is_success, buffer = cv2.imencode(".png", cv_img)
        io_buf = io.BytesIO(buffer)        
        encoded_image = base64.b64encode(buffer).decode("utf-8") 

        messages = [
            {
                "role":"system", 
                "content": [
                    {"type":"text", "text" : "describe a given image"},
                ]
            },
            {
                "role":"user", 
                "content": [
                    {"type":"text", "text" : "Describe the image"},
                    {"type":"image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}#, "detail":"low" }}
                ]
            }
        ] 


        start_time = time.time_ns()
        print(f"{start_time/10**9}")
        model = "gpt-4o-2024-05-13"
        temperature = 0.0
        max_tokens = 256
        results = self.client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        response = []
        for chunk in results:
            response.append(chunk.choices[0].delta.content)
        
        end_time = time.time_ns()
        ans = [m for m in response if m is not None]
        answer = ''.join([m for m in ans])
        print(f"{end_time/10**9}:\tRecieved: {answer}")
        print(f"latency: {(end_time-start_time)/(10**9)}")

if __name__ == '__main__':
    t = Test()

