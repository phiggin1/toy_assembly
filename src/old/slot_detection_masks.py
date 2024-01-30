import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os


def display_img(img):
    # show image
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def distance(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

images = {
    "real_test_image": {"target_x":460, "target_y":375},
    "test":            {"target_x":460, "target_y":375},
    "real_body_test":  {"target_x":900, "target_y":375},
    "image":           {"target_x":700, "target_y":375},
    "image2":          {"target_x":500, "target_y":300},
    "right_frame0000": {"target_x":340, "target_y":140},
    "left_frame0000":  {"target_x":340, "target_y":160}
}

blue = (255, 0, 0)
green = (0,255,0)
red = (0,0,255)
purple = (255,0,128)

slot_thickness = .24*(24) #mm
distance_to_part = 10

min_area = 500

mask_path = os.path.join("test_images", "masks")

files = os.listdir(mask_path)

for file in files:
    print(file)
    if file == ".DS_Stor" or file == ".DS_Store":
        continue

    mask = cv2.imread(os.path.join(mask_path, file), cv2.IMREAD_GRAYSCALE)

    base_filename = file.split('mask')[0][:-1]
    print(base_filename)
    target_x = images[base_filename]["target_x"]
    target_y = images[base_filename]["target_y"]

    #get the contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    print("\tnum contours: "+str(len(contours)))

    for (countour_count,contour) in enumerate(contours):
        contour_img = mask.copy()
        contour_img = cv2.cvtColor(contour_img,cv2.COLOR_GRAY2RGB)

        #display target on image
        cv2.circle(contour_img, (target_x, target_y), radius=3, color=red, thickness=-1)

        #display the contours overlayed on copy of origional image
        cv2.drawContours(contour_img, contours, -1, green, 1)
        
        perimeter = cv2.arcLength(contour,True)
        eps = perimeter*0.0025
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

        '''
        print(contour.shape)
        print(contour)
        display_img(contour_img)
        '''
        hull = cv2.convexHull(contour, returnPoints = False)
        defects = cv2.convexityDefects(contour, hull)
        M = cv2.moments(contour)
        #print(M)
        #get center of the contour
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        #print(cx,cy)
        #cv2.circle(contour_img,(cx,cy),3,blue,-1)

        #print(contour.shape)
        #print(defects.shape[0])
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
                    if d < 30.0:
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
                            if  distance(slot_top, mean) > 30:
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
            #break

        
        
    #break       