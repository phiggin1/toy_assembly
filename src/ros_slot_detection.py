#!/usr/bin/env python3

import math
import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel
from geometry_msgs.msg import PointStamped, Point
from segment_anything import SamPredictor, sam_model_registry

blue = (255, 0, 0)
green = (0,255,0)
red = (0,0,255)
purple = (255,0,128)

def display_img(img):
    # show image
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Unity2Ros(vector3):
    #return (vector3.z, -vector3.x, vector3.y);
    return (vector3[2], -vector3[0], vector3[1])

def distance(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

class SlotTracking:
    def __init__(self):
        rospy.init_node('SlotTracking', anonymous=True)

        self.cam_info_topic = rospy.get_param("cam_info_topic", "/unity/camera/rgb/camera_info")
        self.contour_downsample_amount = rospy.get_param("contour_downsample_amount",0.0025)
        self.slot_width = rospy.get_param("slot_width",3*0.25*24) #slot width in inches (.25in) to mm
        self.slot_height = self.slot_width#0.25*24 #slot height in inches (.25in) to mm

        self.min_area_percentage = rospy.get_param("min_area_percentage", 0.005)
        self.max_area_percentage = rospy.get_param("min_area_percentage", 0.35)
        self.max_num_contours = rospy.get_param("max_num_contours", 12)

        self.image_topic = rospy.get_param("image_topic", "/unity/camera/rgb/image_raw")
        self.location_topic = rospy.get_param("location_topic", "/target_point")


        self.cam_info = rospy.wait_for_message(self.cam_info_topic, CameraInfo)
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(self.cam_info)

        area = self.cam_info.width*self.cam_info.height
        self.min_area = int(self.min_area_percentage*area)
        self.max_area = int(self.max_area_percentage*area)

        rospy.loginfo(f"min area:{self.min_area}\tmax area:{self.max_area}")
        rospy.loginfo(f"slot_width:{self.slot_width}\tslot_height:{self.slot_height}")

        self.image_sub = rospy.wait_for_message(self.image_topic, Image) #rospy.Subscriber(self.image_topic, Image, self.image_cb)
        self.location_sub = rospy.wait_for_message(self.location_topic, PointStamped) #rospy.Subscriber(self.location_topic, PointStamped, self.location_cb)
        
        d = math.sqrt(self.location_sub.point.x**2 + self.location_sub.point.y**2 + self.location_sub.point.z**2)
        p = (self.location_sub.point.x, self.location_sub.point.y, self.location_sub.point.z)
        u, v = self.cam_model.project3dToPixel( p )


        self.cvbridge = CvBridge()
        cv_image = self.cvbridge.imgmsg_to_cv2(self.image_sub, "bgr8")     

        rospy.loginfo('init node')
        self.sam = sam_model_registry["default"](checkpoint="/home/phiggin1/segment-anything/models/sam_vit_h_4b8939.pth")
        self.predictor = SamPredictor(self.sam)
        rospy.loginfo('sam model loaded')

        self.process_image(u,v,d, cv_image)

    def process_image(self, u, v, d, img):
        rospy.loginfo('process start')
        d_mm = d*1000
        # FOV = 2*arctan(x/2f) 
        x = self.cam_info.width
        f = self.cam_model.fx()
        print(d)
        print(x,f)
        half_hfov = math.atan(x/(2*f))/2.0
        print(half_hfov)
        image_half_res_in_mm = d_mm * math.tan(half_hfov)
        pixel_per_mm = x/(image_half_res_in_mm*2)
        slot_width_pixels = self.slot_width*pixel_per_mm
        slot_height_pixels = self.slot_height*pixel_per_mm

        rospy.loginfo(pixel_per_mm)
        rospy.loginfo(f"slot_width_pixels:{slot_width_pixels}\tslot_height_pixels:{slot_height_pixels}")

        target_x = int(u)
        target_y = int(v)

        input_point = np.array([[target_x, target_y]])
        input_label = np.array([1])

        rospy.loginfo('model start')
        
        self.predictor.set_image(img)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        rospy.loginfo('model end')

        for (mask_count,mask) in enumerate(masks):
            imgray = np.asarray(mask*255, dtype=np.uint8)
            #display_img(imgray)
            
            #get the contours
            contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
            
            print(f"mask: {mask_count+1} of {len(masks)}\tnum contours: {len(contours)}")

            if len(contours) > self.max_num_contours:
                continue
            for (countour_count,contour) in enumerate(contours):
                contour_img = imgray.copy()
                contour_img = cv2.cvtColor(contour_img,cv2.COLOR_GRAY2RGB)

                #display target on image
                cv2.circle(contour_img, (target_x, target_y), radius=3, color=red, thickness=-1)

                #display the contours overlayed on copy of origional image
                cv2.drawContours(contour_img, contours, -1, green, 1)
                
                perimeter = cv2.arcLength(contour,True)
                eps = perimeter*self.contour_downsample_amount
                contour2 = cv2.approxPolyDP(contour, eps, closed=True)
                num_points = len(contour2)
                area = cv2.contourArea(contour) 
                if num_points <= 4 or area < self.min_area or area > self.max_area:
                    #print("\tcontour: "+str(countour_count+1)+" of "+str(len(contours))+" too small")
                    continue
                print("\tcontour: "+str(countour_count+1)+" of "+str(len(contours)))
                print("\tcontour area:: "+str(cv2.contourArea(contour)))
                print("\t\t   orig contour len: "+str(len(contour)))
                print("\t\treduced contour len: "+str(len(contour2)))
                contour=contour2

                #draw simlified contour over image
                cv2.drawContours(contour_img, [contour], -1, blue, 2)

                hull = cv2.convexHull(contour, returnPoints = False)
                defects = cv2.convexityDefects(contour, hull)

                print('----------------')
                for k in range(defects.shape[0]):
                    display_image = contour_img.copy()
                    #s - start of convex defect
                    #e - end of convex defect
                    #f - point between start and end
                    #       that is furtherest from convex hull
                    #d - distance of farthest point to hull
                    s,e,f,d = defects[k,0]
                    print(f'Defect:{k}\n\tstart:{s}{contour[s][0]}\n\t  end:{e}{contour[e][0]}')

                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    
                    cv2.line(display_image,start,end,purple,1)
                    #display_img(display_image)   

                    #setup indexes
                    if e > s:
                        indxs = list(range(s,e+1))
                    else:
                        max_indx = contour.shape[0]
                        indxs = list(range(s,max_indx))+list(range(0,e+1))

                    #print(indxs)
                    for array_indx, i in enumerate(indxs): #range(s,e+1):
                        cv2.circle(display_image,tuple(contour[i][0]),3,red,-1)
                        next_contour_indx = array_indx+1
                        last_contour_indx = len(indxs)
                        #for j in range(i+1, e+1):
                        for j in indxs[next_contour_indx:last_contour_indx]:
                            prev = i#(i-1)%len(contour)
                            next = j#(i+1)%len(contour)
                            #distance between start and end of part of defect should be close together (close to slot width)
                            d = distance(tuple(contour[next][0]),tuple(contour[prev][0]))
                            slot_top = tuple(contour[next][0])
                            #print(prev,next,d)
                            if d < slot_width_pixels:
                                #check if next and prev have any points between them
                                if ((prev+1)%contour.shape[0]) != next:
                                    middle = []
                                    for indx in indxs[next_contour_indx:last_contour_indx-1]:     
                                        #print(indx)
                                        middle.append(contour[indx])

                                    middle = np.asarray(middle)
                                    middle = np.reshape(middle, (middle.shape[0],middle.shape[-1]))
                                    
                                    dev = np.std(middle, axis=0)
                                    mean = np.mean(middle, axis=0)
                                    #deviation should be low
                                    #and the base of the slot should be far enough awasy from edge of slot
                                    if  distance(slot_top, mean) > slot_height_pixels:
                                        cv2.circle(display_image,tuple((int(mean[0]),int(mean[1]))),7,purple,-1)
                                        print(prev,next)
                                        print(contour[prev][0], contour[next][0])
                                        #print("middle\n",middle)
                                        print(d)
                                        print("mean:",mean)
                                        print(" dev:",dev)
                                        print(tuple((int(mean[0]),int(mean[1]))))
                                        print('---------------')
                            
                                        #display_img(display_image)   
                                        print('----------------')
            
if __name__ == '__main__':
    track = SlotTracking()
