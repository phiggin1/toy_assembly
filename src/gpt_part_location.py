#!/usr/bin/env python3


import os
import tf
import cv2
import json
import time
import rospy
import base64
import numpy as np
from copy import deepcopy
from openai import OpenAI
from cv_bridge import CvBridge
from multiprocessing import Lock
from copy import deepcopy
from image_geometry import PinholeCameraModel
from toy_assembly.msg import Detection
import pycocotools.mask as mask_util
from toy_assembly.srv import SAM, SAMRequest, SAMResponse
from toy_assembly.srv import ObjectLocation, ObjectLocationRequest, ObjectLocationResponse
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PointStamped

def reject_outliers(data, m=20):
    d = np.abs(data[:,0] - np.median(data[:,0]))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)

    resp = data[s < m]

    return resp

def transform_point(ps, mat44, target_frame):
    xyz = tuple(np.dot(mat44, np.array([ps.point.x, ps.point.y, ps.point.z, 1.0])))[:3]
    r = PointStamped()
    r.header.stamp = ps.header.stamp
    r.header.frame_id = target_frame
    r.point = Point(*xyz)

    return r

def extract_json(text):
    a = text.find('{')
    b = text.rfind('}')+1

    text_json = text[a:b]

    try:
        json_dict = json.loads(text_json)
        return json_dict
    except Exception as inst:
        print(type(inst))    # the exception type
        print(inst.args)     # arguments stored in .args
        print(inst)
        return None

class PartOffset:
    def __init__(self):
        rospy.init_node('part_offset')
        self.cvbridge = CvBridge()
        self.listener = tf.TransformListener()
        self.mutex = Lock()

        self.prefix =  rospy.get_param("~prefix", "test")
        self.debug = rospy.get_param("~debug", True)
        self.sim_time = rospy.get_param("/use_sim_time")#, False)


        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.75
        self.font_color = (255, 255, 255)
        self.font_thickness = 1
        self.line_thickness = 1
        self.alpha = 1.0

        key_filename = rospy.get_param("~key_file", "/home/phiggin1/ai.key")
        with open(key_filename, "rb") as key_file:
            key = key_file.read().decode("utf-8")
            
        self.client = OpenAI(
            api_key = key,
        )

        fp_part_system = "/home/rivr/toy_ws/src/toy_assembly/prompts/gpt_system.txt"
        with open(fp_part_system) as f:
            system = f.read()

        
        self.system = {
            "role":"system", 
            "content": 
            [
                {"type":"text", "text" : system}
            ]
        }

        fp_part_prompt = "/home/rivr/toy_ws/src/toy_assembly/prompts/gpt_part_location.txt"
        with open(fp_part_prompt) as f:
            self.part_prompt = f.read()

        rospy.loginfo(f"sim time active: {self.sim_time}")

        sam_service_name = "/get_sam_segmentation"
        rospy.wait_for_service(sam_service_name)
        self.sam_srv = rospy.ServiceProxy(sam_service_name, SAM)



        self.llm_serv = rospy.Service("/gpt_part_location", ObjectLocation, self.call_gpt)

        rospy.spin()


    def call_gpt(self, req):
        cv_image = self.cvbridge.imgmsg_to_cv2(req.rgb_image)
        depth_image = self.cvbridge.imgmsg_to_cv2(req.depth_image)
        self.cam_info = req.cam_info
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(req.cam_info)        
        target_object = req.object.class_name
        rle = req.object.rle_encoded_mask
        bbox = [req.object.u_min, req.object.v_min, req.object.u_max, req.object.v_max]
        transcript = req.transcription

        print(cv_image.shape)
        print(cv_image.dtype)
        print(depth_image.shape)
        print(depth_image.dtype)

        labeled_image = self.get_labeled_image(cv_image, depth_image, rle, bbox)
        new_msg = self.get_prompt(transcript, labeled_image)
        answer = self.chat_complete(new_msg)

        text = ""
        json_dict = extract_json(answer)
        if "direction" in json_dict:
            text = json_dict["direction"]
        text = text.split(",")

        resp = ObjectLocationResponse()
        resp.directions = text

        return resp

    def get_prompt(self, text, cv_image):
        print("get_prompt")

        is_success, buffer = cv2.imencode(".jpeg", cv_image)
        encoded_image = base64.b64encode(buffer).decode("utf-8") 

        fname = "/home/rivr/encoded_image.jpeg"
        with open(fname, 'wb') as f:
            f.write(buffer)

        instruction = deepcopy(self.part_prompt)
        if instruction.find('[INSTRUCTION]') != -1:
            instruction = instruction.replace('[INSTRUCTION]', text)

        #print(f"==== instr ====\n{instruction}\n==== instr ====")

        prompt_dict = {
            "role":"user", 
            "content": 
            [
                {"type":"text", "text" : instruction},
                {"type":"image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }

        return prompt_dict


    def chat_complete(self, new_msg):
        start_time = time.time_ns()
        
        messages = [ new_msg]
        
        model = "gpt-4o-2024-05-13"
        #model = "gpt-4o-mini-2024-07-18"
        temperature = 0.0
        max_tokens = 750
        
        results = self.client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        response = []
        for chunk in results:
            response.append(chunk.choices[0].delta.content)

        ans = [m for m in response if m is not None]
        answer = ''.join([m for m in ans])

        end_time = time.time_ns()

        rospy.loginfo(f"GPT resp:\n {answer}\n")
        rospy.loginfo(f"GPT resp latency: {(end_time-start_time)/(10**9)}")

        return answer
    
    def get_labeled_image(self, cv_image, depth_image, rle, bbox):
        #get the position and mask for the object
        center, mask = self.get_position(json.loads(rle), bbox, depth_image)
        masked_color = cv2.bitwise_and(cv_image, cv_image, mask=mask)

        rospy.loginfo(f"position x:{center.point.x:.2f}, y:{center.point.y:.2f}, z:{center.point.z:.2f}")

        im_gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
        min_x,min_y,w,h = cv2.boundingRect(im_gray)

        print(w,h,w*h)
        if w>125:
            num_horiz = 3
        elif w>50:
            num_horiz = 2
        else:
            num_horiz = 1

        if h>125:
            num_vert = 2
        else:
            num_vert = 1

        x_vals = w*(np.asarray(range(0,num_horiz+1))/num_horiz)+min_x
        y_vals = h*(np.asarray(range(0,num_vert+1))/num_vert)+min_y
        colors = [
            (255, 255,   0),
            (  0, 255, 255),
            (255,   0, 255),
            (255,   0,   0),
            (  0, 255,   0),
            (  0,   0, 255)
        ]
        rect_img = masked_color.copy()
        cnt = 0
        for i in range(len(x_vals)-1):
            for j in range(len(y_vals)-1):
                start_point = (int(x_vals[i]) , int(y_vals[j]))
                end_point =   (int(x_vals[i+1]),int(y_vals[j+1]))
                rect_img = cv2.rectangle(rect_img, start_point, end_point, colors[cnt], -1)
                cnt += 1

        rect_img = cv2.bitwise_and(rect_img, rect_img, mask=mask)
        beta = 1.0-self.alpha
        masked_color = cv2.addWeighted(masked_color, beta, rect_img, self.alpha, 0.0)

        im_gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
        x,y,w,h = cv2.boundingRect(im_gray)
        x_max = x + w
        y_max = y + h
        masked_color = masked_color[y:y_max,x:x_max]
        
        #print(len(np.unique(masked_color.reshape(-1, masked_color.shape[2]), axis=0)))
        for l in np.unique(masked_color.reshape(-1, masked_color.shape[2]), axis=0):
            imgray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(masked_color, contours, -1, (0,255,0), 3)
            
        #length of the axis lines to draw
        offset = 0.15

        front = deepcopy(center)
        front.point.x -= offset
        back = deepcopy(center)
        back.point.x += offset
        right = deepcopy(center)
        right.point.y -= offset
        left = deepcopy(center)
        left.point.y += offset
        above = deepcopy(center)
        above.point.z += offset
        below = deepcopy(center)
        below.point.z -= offset
    
        lines = [
            [front, back,  "Front", "Back",  (255, 0,   0  )],
            [right, left,  "Right", "left",  (0,   255, 0  )],
            [above, below, "Above", "Below", (0,   0,   255)],
        ]

        translation,rotation = self.listener.lookupTransform(self.cam_info.header.frame_id, 'world', rospy.Time(0))
        mat44 = self.listener.fromTranslationRotation(translation, rotation)

        #draw the x,y,z axis lines
        for line in lines:
            a_u,a_v = self.project_to_pixel(line[0], mat44, self.cam_info.header.frame_id)
            b_u,b_v = self.project_to_pixel(line[1], mat44, self.cam_info.header.frame_id)
            cv2.line(masked_color, (a_u, a_v), (b_u, b_v), line[4], self.line_thickness) 

        #Add in the text labels for each axis
        for line in lines:
            a_u,a_v = self.project_to_pixel(line[0], mat44, self.cam_info.header.frame_id)
            size, baseline = cv2.getTextSize(line[2], self.font, self.font_scale, self.font_thickness)
            a_u = max(min(a_u, masked_color.shape[1]-size[0]), 0)
            a_v = max(min(a_v, masked_color.shape[0]-size[1]), 0)
            cv2.putText(masked_color, line[2], (a_u, a_v), self.font, self.font_scale, self.font_color, self.font_thickness, cv2.LINE_AA)

            b_u,b_v = self.project_to_pixel(line[1], mat44, self.cam_info.header.frame_id)
            size, baseline = cv2.getTextSize(line[3], self.font, self.font_scale, self.font_thickness)
            b_u = max(min(b_u, masked_color.shape[1]-size[0]), 0)
            b_v = max(min(b_v, masked_color.shape[0]-size[1]), 0)
            cv2.putText(masked_color, line[3], (b_u, b_v), self.font, self.font_scale, self.font_color, self.font_thickness, cv2.LINE_AA)
        
        return masked_color

    def project_to_pixel(self, pt, mat44, target_frame):
        a = transform_point(pt, mat44, target_frame)
        a_list = [a.point.x,a.point.y,a.point.z]
        u, v  = self.cam_model.project3dToPixel(a_list)

        u = int(u)
        v = int(v)

        return (u,v)


   
    def get_position(self, rle, bbox, depth_image):
        #get the mask from the compressed rle
        # get the image coords of the bounding box
        mask = mask_util.decode([rle])
        u_min = int(bbox[0])
        v_min = int(bbox[1])
        u_max = int(bbox[2])
        v_max = int(bbox[3])

        translation,rotation = self.listener.lookupTransform('world', self.cam_info.header.frame_id, rospy.Time(0))
        mat44 = self.listener.fromTranslationRotation(translation, rotation)

        #get the camera center (cx,cy) and focal length (fx,fy)
        cx = self.cam_model.cx()
        cy = self.cam_model.cy()
        fx = self.cam_model.fx()
        fy = self.cam_model.fy()

        distances = []
        for v in range(v_min, v_max, 1):
            for u in range(u_min, u_max, 1):
                if mask[v][u] > 0:
                    d = (depth_image[v][u])
                    #if depth is 16bit int the depth value is in mm
                    # convert to m
                    if isinstance(d, np.uint16):
                        d = d/1000.0
                    #toss out points that are too close or too far 
                    # to simplify outlier rejection
                    if d <= 0.1 or d > 1.5 or np.isnan(d):
                        continue
                    distances.append((d, v, u))

        #Filter out outliers
        if len(distances)>0:
            distances = reject_outliers(np.asarray(distances), m=2)

        #transform all the points into the world frame
        # and get the center
        x = 0
        y = 0
        z = 0
        for dist in distances:
            d = dist[0]
            v = dist[1]
            u = dist[2]
            ps = PointStamped()
            ps.header.frame_id = self.cam_info.header.frame_id
            ps.point.x = (u - cx)*d/fx
            ps.point.y = (v - cy)*d/fy
            ps.point.z = d
            t_p = transform_point(ps, mat44, 'world')
            x += t_p.point.x
            y += t_p.point.y
            z += t_p.point.z
        
        center = PointStamped()
        center.header.frame_id = 'world'
        center.point.x = x/len(distances)
        center.point.y = y/len(distances)
        center.point.z = z/len(distances)

        return center, mask
    
if __name__ == '__main__':
    llm = PartOffset()