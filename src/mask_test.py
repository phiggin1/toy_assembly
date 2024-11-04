#!/usr/bin/env python3


import os
import tf
import cv2
import json
import numpy as np
import rospy
import message_filters
from cv_bridge import CvBridge
from multiprocessing import Lock
from copy import deepcopy
from image_geometry import PinholeCameraModel
from toy_assembly.msg import Detection
import pycocotools.mask as mask_util
from toy_assembly.srv import SAM, SAMRequest, SAMResponse
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

class AssemblyClient:
    def __init__(self):
        rospy.init_node('test')
        self.cvbridge = CvBridge()
        self.listener = tf.TransformListener()
        self.mutex = Lock()

        self.prefix =  rospy.get_param("~prefix", "test")
        self.debug = rospy.get_param("~debug", True)
        self.sim_time = rospy.get_param("/use_sim_time")#, False)

        rospy.loginfo(f"sim time active: {self.sim_time}")
        if not self.sim_time:
            self.cam_link_name = f"left_camera_color_frame"
            cam_info_topic = f"/left_camera/color/camera_info_throttled"
            rgb_image_topic = f"/left_camera/color/image_rect_color_throttled"              
            depth_image_topic = f"/left_camera/depth_registered/sw_registered/image_rect_throttled"
        else:
            self.cam_link_name = f"left_camera_link"
            cam_info_topic = f"/unity/camera/left/rgb/camera_info"
            rgb_image_topic = f"/unity/camera/left/rgb/image_raw"
            depth_image_topic = f"/unity/camera/left/depth/image_raw"

        rospy.loginfo(cam_info_topic)
        rospy.loginfo(rgb_image_topic)
        rospy.loginfo(depth_image_topic)

        sam_service_name = "/get_sam_segmentation"
        rospy.wait_for_service(sam_service_name)
        self.sam_srv = rospy.ServiceProxy(sam_service_name, SAM)

        self.cam_info = rospy.wait_for_message(cam_info_topic, CameraInfo)
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(self.cam_info)

        self.rgb_sub = message_filters.Subscriber(rgb_image_topic, Image)
        self.depth_sub = message_filters.Subscriber(depth_image_topic, Image)
        ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], slop=0.2, queue_size=10)
        ts.registerCallback(self.image_cb)

        self.rgb_image = None
        self.depth_image = None
        self.have_images = False
        
        rate = rospy.Rate(5)
        while not self.have_images:
            rate.sleep()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        font_color = (255, 255, 255)
        font_thickness = 1
        line_thickness = 1

        translation,rotation = self.listener.lookupTransform(self.cam_info.header.frame_id, 'world', rospy.Time(0))
        mat44 = self.listener.fromTranslationRotation(translation, rotation)
        #image, objects, rles, bboxs, scores = self.get_detections("tan tray. orange tray. tan horse body. blue horse legs. orange horse legs. table. robot gripper.")
        #image, objects, rles, bboxs, scores = self.get_detections("blue legs.")
        image, objects, rles, bboxs, scores = self.get_detections("horse head.")

        with self.mutex:
            cv_image = self.rgb_image.copy()

        for i, o in enumerate(objects):
            #get the position and mask for the object
            center, mask = self.get_position(json.loads(rles[i]), bboxs[i])
            masked_color = cv2.bitwise_and(cv_image, cv_image, mask=mask)

            rospy.loginfo(f"{i}, {o} at position x:{center.point.x:.2f}, y:{center.point.y:.2f}, z:{center.point.z:.2f}")

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
            alpha = 1.0
            beta = 1.0-alpha
            masked_color = cv2.addWeighted(masked_color, beta, rect_img, alpha, 0.0)

            
            im_gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
            x,y,w,h = cv2.boundingRect(im_gray)
            x_max = x + w
            y_max = y + h
            masked_color = masked_color[y:y_max,x:x_max]
            
            print(len(np.unique(masked_color.reshape(-1, masked_color.shape[2]), axis=0)))
            for l in np.unique(masked_color.reshape(-1, masked_color.shape[2]), axis=0):
                print(l)
                imgray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
                contours, hierarchy = cv2.findContours(imgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(masked_color, contours, -1, (0,255,0), 3)
                print(len(contours))
            # Display the image in a window named 'Image'
            cv2.imshow(o, masked_color)
            # Wait for a key press indefinitely or for a specified amount of time in milliseconds
            cv2.waitKey(0)
            # Close all OpenCV windows
            cv2.destroyAllWindows()
            
            filename = "/home/rivr/masked_seg_axis.png"
            cv2.imwrite(filename, masked_color)
            '''
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
                [front, back,  "front", "back",  (255,0,0)],
                [right, left,  "right", "left",  (0,255,0)],
                [above, below, "above", "below", (0,0,255)],
            ]
            #draw the x,y,z axis lines
            for line in lines:
                a_u,a_v = self.project_to_pixel(line[0], mat44, self.cam_info.header.frame_id)
                b_u,b_v = self.project_to_pixel(line[1], mat44, self.cam_info.header.frame_id)
                cv2.line(masked_color, (a_u, a_v), (b_u, b_v), line[4], line_thickness) 

            #Add in the text labels for each axis
            for line in lines:
                a_u,a_v = self.project_to_pixel(line[0], mat44, self.cam_info.header.frame_id)
                size, baseline = cv2.getTextSize(line[2], font, font_scale, font_thickness)
                a_u = max(min(a_u, masked_color.shape[1]-size[0]), 0)
                a_v = max(min(a_v, masked_color.shape[0]-size[1]), 0)
                cv2.putText(masked_color, line[2], (a_u, a_v), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

                b_u,b_v = self.project_to_pixel(line[1], mat44, self.cam_info.header.frame_id)
                size, baseline = cv2.getTextSize(line[3], font, font_scale, font_thickness)
                b_u = max(min(b_u, masked_color.shape[1]-size[0]), 0)
                b_v = max(min(b_v, masked_color.shape[0]-size[1]), 0)
                cv2.putText(masked_color, line[3], (b_u, b_v), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            '''


        return

    def project_to_pixel(self, pt, mat44, target_frame):
        a = transform_point(pt, mat44, target_frame)
        a_list = [a.point.x,a.point.y,a.point.z]
        u, v  = self.cam_model.project3dToPixel(a_list)

        u = int(u)
        v = int(v)

        return (u,v)

    def image_cb(self, rgb_ros_image, depth_ros_image):
        success = self.mutex.acquire(block=False, timeout=0.02)
        if success:
            #print(f"rgb eoncoding: {rgb_ros_image.encoding}")
            #print(f"depth eoncoding: {depth_ros_image.encoding}")
            self.rgb_encoding = rgb_ros_image.encoding
            self.depth_encoding = depth_ros_image.encoding
            self.rgb_image = self.cvbridge.imgmsg_to_cv2(rgb_ros_image, desired_encoding="bgr8") 
            self.depth_image = self.cvbridge.imgmsg_to_cv2(depth_ros_image)
            if not self.have_images:
                rospy.loginfo("HAVE IMAGES")
                self.have_images = True
            self.mutex.release()

    def get_detections(self, text):
        req = SAMRequest()
        req.text_prompt = text
        resp = self.sam_srv(req)

        annotated_img = resp.annotated_image
        
        objects = []
        rles = []
        bboxes = []
        scores = []
        for obj in resp.object:
            objects.append(obj.class_name)
            rles.append(obj.rle_encoded_mask)
            box = [obj.u_min, obj.v_min,obj.u_max, obj.v_max]
            bboxes.append(box)
            scores.append(obj.score)

        return (annotated_img, objects, rles, bboxes, scores)
    
    def get_position(self, rle, bbox):
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
                    d = (self.depth_image[v][u])
                    #if depth is 16bit int the depth value is in mm
                    # convert to m
                    if self.depth_encoding == "16UC1":
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
    llm = AssemblyClient()