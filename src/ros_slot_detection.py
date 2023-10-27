#!/usr/bin/env python3

import rospy
import math
import io
import math
import numpy as np
from cv_bridge import CvBridge
from segment_anything import SamPredictor, sam_model_registry

CAMERA = "/gen3_robotiq_2f_85_left/world/base_link/shoulder_link/bicep_link/forearm_link/spherical_wrist_1_link/spherical_wrist_2_link/bracelet_link/end_effector_link/camera_link/camera_standin"
OBJECT = "/horse_body (2)"

def display_img(img):
    # show image
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Unity2Ros(vector3):
    #return (vector3.z, -vector3.x, vector3.y);
    return (vector3[2], -vector3[0], vector3[1])

class SlotTracking:
    def __init__(self):
        rospy.init_node('SlotTracking', anonymous=True)

        self.sam = sam_model_registry["default"](checkpoint="../../sam_vit_h_4b8939.pth")
        self.predictor = SamPredictor(sam)

        self.contour_downsample_amount = 0.0025
        self.slot_width = 0.25/24 #slot width in inches (.25in) to mm
        self.min_area = 500


        self.cam_info = rospy.wait_for_message('/camera/rgb/camera_info', CameraInfo)
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(self.cam_info)

        self.image_sub = rospy.wait_for_message('/camera/rgb/image_raw', Image) #rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_cb)
        self.location_sub = rospy.wait_for_message('target_point', PointStamped) #rospy.Subscriber("/target_point", PointStamped, self.location_cb)
        d = math.sqrt(self.location_sub.point.x**2 + self.location_sub.point.y**2 + self.location_sub.point.z**2)
        u, v = self.cam_model.project3dToPixel( self.location_sub )

        self.cvbridge = CvBridge()
        cv_image = self.cvbridge.imgmsg_to_cv2(self.image_sub, "bgr8")     
        self.d_mm(u,v,d, cv_image)

    def process_image(self, u, v, d, img):
        d_mm = d*1000
        # FOV = 2*arctan(x/2f) 
        x = self.camera_info.width
        f = self.camera_model.fx
        half_hfov = math.atan(x/(2*f))
        half_hres = x/2
        pixel_per_mm = d * math.tan(half_hres)
        slot_width_pixels = self.slot_width*pixel_per_mm
        slot_height_pixels = self.slot_height*pixel_per_mm

        target_x = u
        target_y = v
        self.predictor.set_image(img)
        input_point = np.array([[target_x, target_y]])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        for (mask_count,mask) in enumerate(masks):
            print("mask: "+str(mask_count+1)+" of "+str(len(masks)))
            imgray = np.asarray(mask*255, dtype=np.uint8)


            #get the contours
            contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
            print("\tnum contours: "+str(len(contours)))

            for (countour_count,contour) in enumerate(contours):
                contour_img = mask.copy()
                contour_img = cv2.cvtColor(contour_img,cv2.COLOR_GRAY2RGB)

                #display target on image
                cv2.circle(contour_img, (target_x, target_y), radius=3, color=red, thickness=-1)

                #display the contours overlayed on copy of origional image
                cv2.drawContours(contour_img, contours, -1, green, 1)
                
                perimeter = cv2.arcLength(contour,True)
                eps = perimeter*self.contour_downsample_amount
                contour2 = cv2.approxPolyDP(contour, eps, closed=True)

                num_points = len(contour2)
                if num_points <= 4 or cv2.contourArea(contour) < min_area:
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
                    far = tuple(contour[f][0])
                    
                    cv2.line(display_image,start,end,purple,1)

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
                            
                                        display_img(display_image)   
                                        print('----------------')

if __name__ == '__main__':
    c = control_image()
