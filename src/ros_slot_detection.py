#!/usr/bin/env python3

import math
import numpy as np
import cv2
import rospy
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel
from geometry_msgs.msg import PointStamped, Point
from toy_assembly.srv import SAM
from toy_assembly.srv import DetectSlot, DetectSlotResponse


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
                
        '''
        self.cam_info_topic =    rospy.get_param("cam_info_topic",  "/unity/camera/rgb/camera_info")
        self.rgb_image_topic =   rospy.get_param("rgb_image_topic",     "/unity/camera/rgb/image_raw")
        self.depth_image_topic = rospy.get_param("depth_image_topic",     "/unity/camera/depth/image_raw")
        self.location_topic =    rospy.get_param("location_topic",  "/pt")
        self.cam_info = rospy.wait_for_message(self.cam_info_topic, CameraInfo)
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(self.cam_info)

        area = self.cam_info.width*self.cam_info.height
        self.min_area = int(self.min_area_percentage*area)
        self.max_area = int(self.max_area_percentage*area)

        rospy.loginfo(f"slot width:{self.slot_width}mm\tslot height:{self.slot_height}mm")
        rospy.loginfo(f"min area:{self.min_area}\tmax area:{self.max_area}")
        '''
        self.contour_downsample_amount = rospy.get_param("contour_downsample_amount",0.0025)
        self.slot_width = rospy.get_param("slot_width",3*0.25*24) #slot width in inches (.25in) to mm
        self.slot_height = self.slot_width#0.25*24 #slot height in inches (.25in) to mm
        self.min_area_percentage = rospy.get_param("min_area_percentage", 0.005)
        self.max_area_percentage = rospy.get_param("min_area_percentage", 0.35)
        self.max_num_contours = rospy.get_param("max_num_contours", 12)

        rospy.loginfo(f"slot width:{self.slot_width}mm\tslot height:{self.slot_height}mm")
        rospy.loginfo(f"min_area_percentage:{self.min_area_percentage}\tmax_area_percentage:{self.max_area_percentage}")
        rospy.loginfo(f"max_num_contours:{self.max_num_contours}")

        self.listener = tf.TransformListener()
        self.cvbridge = CvBridge()

        self.detect_slot_serv = rospy.Service('get_slot_location', DetectSlot, self.detect_slot)
        self.segment_serv = rospy.ServiceProxy('get_sam_segmentation', SAM)

        rospy.spin()


        #d = 0.1
        #u = 200
        #v = 200

        #cv_image = self.cvbridge.imgmsg_to_cv2(self.rgb_image_sub, "bgr8")     
        #cv_depth = np.asarray(self.cvbridge.imgmsg_to_cv2(self.depth_image_sub, desired_encoding="passthrough"))
        #cv_depth = None
        

        
        #org self.process_image(u,v,d, cv_image, cv_depth)
        #self.process_image(u,v,d, self.rgb_image_sub, cv_depth)

    def transform_point(self, obj_position, target_frame):
        t = rospy.Time.now()
        obj_position.header.stamp = t
        self.listener.waitForTransform(obj_position.header.frame_id, target_frame, t, rospy.Duration(4.0))
        transformed_position = self.listener.transformPoint(target_frame, obj_position)
        
        return transformed_position
    
    def detect_slot(self, req):
        rgb_image = req.rgb_image
        depth_image = req.depth_image
        cam_info = req.cam_info
        location = req.location

        location = self.transform_point(location, cam_info.header.frame_id)

        cam_model = PinholeCameraModel()
        cam_model.fromCameraInfo(cam_info)

        area = cam_info.width*cam_info.height
        self.min_area = int(self.min_area_percentage*area)
        self.max_area = int(self.max_area_percentage*area)

        rospy.loginfo(f"min area:{self.min_area}\tmax area:{self.max_area}")


        d = math.sqrt(location.point.x**2 + location.point.y**2 + location.point.z**2)
        p = (location.point.x, location.point.y, location.point.z)

        rospy.loginfo(p)

        u, v = cam_model.project3dToPixel( p )
        u = int(u)
        v = int(v)

        display_imgage = self.cvbridge.imgmsg_to_cv2(rgb_image).copy()
        cv2.circle(display_imgage,tuple((u,v)),7,purple,-1)
        #display_img(display_imgage)

        d_mm = d*1000
        # FOV = 2*arctan(x/2f) 
        x = cam_info.width
        f = cam_model.fx()

        half_hfov = math.atan(x/(2*f))/2.0
        image_half_res_in_mm = d_mm * math.tan(half_hfov)
        pixel_per_mm = x/(image_half_res_in_mm*2)
        slot_width_pixels = self.slot_width*pixel_per_mm
        slot_height_pixels = self.slot_height*pixel_per_mm
        target_x = int(u)
        target_y = int(v)

        rospy.loginfo(f"d:{d}")
        rospy.loginfo(f"x:{x}, f:{f}")
        rospy.loginfo(f"half_hfov:{half_hfov}")
        rospy.loginfo(f"image_half_res_in_mm{image_half_res_in_mm}")
        rospy.loginfo(f"pixel per mm: {pixel_per_mm}")
        rospy.loginfo(f"slot_width_pixels: {slot_width_pixels}")
        rospy.loginfo(f"slot_height_pixels: {slot_height_pixels}")
        rospy.loginfo(f"u:{u}, v:{v}")
        rospy.loginfo(f"target_x:{target_x}, target_y:{target_y}")


        slots = []
        #org masks = self.serv(self.cvbridge.cv2_to_imgmsg(img, "bgr8"), target_x, target_y)
        resp  = self.segment_serv(rgb_image, target_x, target_y)
        print(len(resp.masks))
        masks = resp.masks
        for (mask_count,mask) in enumerate(masks):
            mask = self.cvbridge.imgmsg_to_cv2(mask)
            imgray = np.asarray(mask*255, dtype=np.uint8)
            #display_img(imgray)
            rospy.loginfo(f"mask: {mask_count+1} of {len(masks)}")
            
            '''
            depth_masked = cv2.bitwise_and(d_img, d_img, mask=mask.astype(np.uint8))
            display_img(depth_masked)
            
            #get mm per pixel here
            
            #get points around target_x,targey_y +-5
            #get positions of points 
            #filter outliers
            #get mean delta_d between points
            
            rospy.loginfo(np.min(depth_masked))
            rospy.loginfo(np.max(depth_masked))
            window = 5
            min_x = target_x - window
            min_y = target_y - window
            max_x = target_x + window
            max_y = target_y + window
            points = []
            for x in range(min_x,max_x):
                for y in range(min_y,max_y):
                    if depth_masked[x,y] > 0.0:
                        #rospy.loginfo(depth_masked[x,y])
                        points.append((x,y,depth_masked[x,y]))
            rospy.loginfo(points)
            self.process_mask(imgray, d_img, target_x, target_y, slot_width_pixels, slot_height_pixels)
            '''
            
            slot = self.process_mask(imgray, target_x, target_y, slot_width_pixels, slot_height_pixels)
            if slot is not None:
                for s in slot:
                    slots.append(s)

        #convert from image space to 3d space
        #use uv from slots + d/d_mm
        slot_locations = []
        for s in slots:
            rospy.loginfo(f"d:{d},s{s}")
            p = cam_model.projectPixelTo3dRay(s)
            rospy.loginfo(p)
            stamped_point = PointStamped()
            stamped_point.point.x = p[0]*d
            stamped_point.point.y = p[1]*d
            stamped_point.point.z = p[2]*d
            stamped_point.header = rgb_image.header
            slot_locations.append(stamped_point)

        resp = DetectSlotResponse()
        resp.slot_locations = slot_locations

        return resp

    #def process_mask(self, imgray, d_img, target_x, target_y, slot_width_pixels, slot_height_pixels):
    def process_mask(self, imgray, target_x, target_y, slot_width_pixels, slot_height_pixels):
        #get the contours
        contours, _ = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        if len(contours) > self.max_num_contours:
            rospy.loginfo(f"{len(contours)}>{self.max_num_contours}")
            return None
        
        slots = []
        for (countour_count,contour) in enumerate(contours):
            rospy.loginfo(f"contour: {countour_count+1} of {len(contours)}")
            '''
            result = cv2.pointPolygonTest(contour, (target_x,target_y), False)
            rospy.loginfo(result)
            if result == 0:
                rospy.loginfo("on countour")
            elif result > 0:
                rospy.loginfo("in countour")
            elif result < 0:
                rospy.loginfo("outside countour")
            '''
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
                rospy.loginfo("\tcontour: "+str(countour_count+1)+" of "+str(len(contours))+" too small")
                continue
            rospy.loginfo(f"\tcontour: {countour_count+1} of {len(contours)}")
            rospy.loginfo(f"\tcontour area:: {cv2.contourArea(contour)}")
            rospy.loginfo(f"\t\t   orig contour len: {len(contour)}")
            rospy.loginfo(f"\t\treduced contour len: {len(contour2)}")
            contour=contour2

            #draw simlified contour over image
            cv2.drawContours(contour_img, [contour], -1, blue, 2)
            hull = cv2.convexHull(contour, returnPoints = False)
            defects = cv2.convexityDefects(contour, hull)

            rospy.loginfo('----------------')
            for k in range(defects.shape[0]):
                slot = []
                display_image = contour_img.copy()
                #s - start of convex defect
                #e - end of convex defect
                #f - point between start and end that is furtherest from convex hull
                #d - distance of farthest point to hull
                s,e,_,_ = defects[k,0]
                rospy.loginfo(f'Defect:{k}\n\tstart:{s}{contour[s][0]}\n\t  end:{e}{contour[e][0]}')
                
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                
                cv2.line(display_image,start,end,purple,1)

                #setup indexes
                if e > s:
                    indxs = list(range(s,e+1))
                else:
                    max_indx = contour.shape[0]
                    indxs = list(range(s,max_indx))+list(range(0,e+1))

                #rospy.loginfo(indxs)
                #rospy.loginfo(contour[indxs])
                #rospy.loginfo(len(indxs))
                for i in indxs: #range(s,e+1):
                    cv2.circle(display_image,tuple(contour[i][0]),3,red,-1)
                #display_img(display_image)   

                for i in range(len(indxs)):
                    for j in range(i+2,len(indxs)):
                        #rospy.loginfo(f"i:{i}, contour index:{indxs[i]} point:{contour[indxs[i]]}")
                        #rospy.loginfo(f"j:{j}, contour index:{indxs[j]} point:{contour[indxs[j]]}")
                        prev = indxs[i]
                        next = indxs[j]
                        d = distance(tuple(contour[next][0]),tuple(contour[prev][0]))
                        slot_top = tuple(contour[next][0])
                       
                        slot_start = contour[prev][0]
                        slot_end = contour[next][0]
                        a = int((slot_start[0]+slot_end[0])/2.0)
                        b = int((slot_start[1]+slot_end[1])/2.0)
                        slot_top = tuple([a,b])

                        #rospy.loginfo(f"{i}, {j}, {prev}, {next}, {d}")
                        if d < slot_width_pixels:
                            #check if next and prev have any points between them
                            if ((prev+1)%contour.shape[0]) < next:
                                middle = []
                               
                                rospy.loginfo(f"i:{i}, j:{j} prev:{prev}, next:{next}")
                                for indx in indxs[i:j-1]:     
                                    middle.append(contour[indx])
                                middle = np.asarray(middle)
                                middle = np.reshape(middle, (middle.shape[0],middle.shape[-1]))                               
                                dev = np.std(middle, axis=0)
                                mean = np.mean(middle, axis=0)
                                potential_lot_height = distance(slot_top, mean)

                                rospy.loginfo(f"middle:{middle}")
                                rospy.loginfo(f"mean:{mean}")
                                rospy.loginfo(f"dev:{dev}")
                                rospy.loginfo(f"slot_top:{slot_top}")
                                rospy.loginfo(f"potential slot height:{potential_lot_height}")

                                #deviation should be low
                                #and the 'base' of the slot should be far enough awasy from opening of slot
                                if potential_lot_height > slot_height_pixels:
                                    #cv2.circle(display_image,tuple(slot_start),4,green,-1)
                                    #cv2.circle(display_image,tuple(slot_end),4,green,-1)
                                    #cv2.circle(display_image,tuple(slot_top),4,green,-1)
                                    #cv2.circle(display_image,tuple((int(mean[0]),int(mean[1]))),7,purple,-1)
                                    #rospy.loginfo(prev,next)
                                    #rospy.loginfo(contour[prev][0], contour[next][0])
                                    #rospy.loginfo("middle\n",middle)
                                    #rospy.loginfo("mean:",mean)
                                    #rospy.loginfo(" dev:",dev)
                                    #rospy.loginfo('----------------')
                                    rospy.loginfo(tuple((int(mean[0]),int(mean[1]))))
                                    slot.append(tuple((int(mean[0]),int(mean[1]))))
                                    #display_img(display_image)   
                                    rospy.loginfo('----------------')

                rospy.loginfo(f"slots:{slot}")
                for s in slot:
                    cv2.circle(display_image,tuple((s[0],s[1])),7,purple,-1)
                    slots.append(s)
                #display_img(display_image)

        rospy.loginfo(f"slots:{slots}")

        return slots
       
if __name__ == '__main__':
    track = SlotTracking()
