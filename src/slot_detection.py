import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from segment_anything import SamPredictor, sam_model_registry

def display_img(img):
    # show image
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



sam = sam_model_registry["default"](checkpoint="../../sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

image_dir  = 'test_images'
angle_threshold = 0.3
pixel_distance_threshold = 15
min_area = 250



images = [
    {"filename":'real_test_image.jpg',  "target_x":460, "target_y":375},
    {"filename":'test.jpg',             "target_x":460, "target_y":375},
    {"filename":'real_body_test.jpg',   "target_x":900, "target_y":375},
    {"filename":'image.png',            "target_x":700, "target_y":375},
    {"filename":'image2.png',           "target_x":500, "target_y":300},
    {"filename":'right_frame0000.jpg',  "target_x":340, "target_y":140},
    {"filename":'left_frame0000.jpg',   "target_x":340, "target_y":160}
]


for image in images:
    image_path = image["filename"]
    target_x = image["target_x"]
    target_y = image["target_y"]
    print('------------')
    print(os.path.join(image_dir, image_path))

    img = cv2.imread(os.path.join(image_dir, image_path))

    '''
    disp_img = img.copy()
    cv2.circle(disp_img, (target_x, target_y), radius=3, color=(0, 0, 255), thickness=-1)
    display_img(disp_img)
    '''

    predictor.set_image(img)
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
        #display_img(imgray)


        image_info = "mask_"+str(mask_count+1)
        fname, extension = os.path.splitext(image_path)
        outfilename = fname+"_"+image_info+".png"#extension
        out_path = os.path.join(image_dir,"masks", outfilename)
        cv2.imwrite(out_path, imgray)

        #get the contours
        contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        print("\tnum contours: "+str(len(contours)))

        #if there are a lot ofs contours the segmentation did something wierd 
        # so skip the mask
        if len(contours)>5:
            continue
        
        for (countour_count,contour) in enumerate(contours):
            contour_img = img.copy()

            #display target on image
            cv2.circle(contour_img, (target_x, target_y), radius=3, color=(255, 255, 255), thickness=-1)

            print("\tcontour: "+str(countour_count+1)+" of "+str(len(contours)))
            
            #display the contours overlayed on copy of origional image
            cv2.drawContours(contour_img, contours, -1, (0,255,0), 1)
            
            perimeter = cv2.arcLength(contour,True)
            eps = perimeter*0.01
            contour2 = cv2.approxPolyDP(contour, eps, closed=True)
            print("\t\t   orig contour len: "+str(len(contour)))
            print("\t\treduced contour len: "+str(len(contour2)))
            contour=contour2

            #draw simlified contour over image
            cv2.drawContours(contour_img, [contour], -1, (255,255,255), 1)

            right_angles = []
            slots = []
            num_points = len(contour)
            print("\t\tnum points: "+str(num_points))

            #print(cv2.contourArea(contour))
            if num_points <= 4 or cv2.contourArea(contour) < min_area:
                print("\t\tcontour is too small")
                continue

            for i in range(num_points):
                #get the next 4 points (wrapping around the end of the list back to the beginning)
                p0 = contour[i%num_points][0]
                p1 = contour[(i+1)%num_points][0]
                p2 = contour[(i+2)%num_points][0]
                p3 = contour[(i+3)%num_points][0]

                #draw the point on the image
                cv2.circle(contour_img, (p0[0], p0[1]), radius=2, color=(0, 0, 255), thickness=-1)

                #for the next four points get the 3 vectors between them (0 to 1, 1 to 2, 2 to 3) 
                v1 = p1-p0
                v2 = p2-p1
                v3 = p3-p2
                v4 = p0-p3

                #calculate the angle between each sequential pair of vectors
                cos_theta1 = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
                cos_theta2 = np.dot(v2,v3)/(np.linalg.norm(v2)*np.linalg.norm(v3))
                cos_theta3 = np.dot(v3,v4)/(np.linalg.norm(v3)*np.linalg.norm(v4))

                


                '''
                print(p0[0],p0[1],
                      v1[0],v1[1],
                      v2[0],v2[1],
                      v3[0],v3[1],
                      v4[0],v4[1], 
                      cos_theta1,cos_theta2,cos_theta3)
                '''
                

                #check that the 4 points are at ~90deg apart 
                if abs(cos_theta1) < angle_threshold and abs(cos_theta2) < angle_threshold and abs(cos_theta3 < angle_threshold):
                        # and not too close together
                        #if cos_theta1*cos_theta2 > 0 and cos_theta2*cos_theta3 > 0:
                        #if np.linalg.norm(v1[0]) > pixel_distance_threshold and np.linalg.norm(v2[0]) > pixel_distance_threshold and np.linalg.norm(v3[0]) > pixel_distance_threshold:
                        print(np.linalg.norm(v4),np.linalg.norm(v2))
                        print(abs(np.linalg.norm(v4)-np.linalg.norm(v2)))
                        if abs(np.linalg.norm(v4)-np.linalg.norm(v2))<30:
                            #print(cos_theta1)
                            #print(cos_theta2)
                            #print('--------')
                            right_angles.append([p0.tolist()])
                            right_angles.append([p1.tolist()])
                            right_angles.append([p2.tolist()])
                            right_angles.append([p3.tolist()])
                            slots.append( [p0,p1,p2,p3] )
           
            #draw the points making up the slots on the image
            right_angles = np.asarray(right_angles)

            slots = np.asarray(slots)


            
            print("\tnum slots: "+str(len(slots)))
            if len(slots) < 1:
                print("\tno slots found")
            else:
                for slot in slots:
                    print("\t"+np.array2string(slot, separator=','))
            
            cv2.drawContours(contour_img, right_angles,  -1, (255, 0, 0), 8)
            print("\tfinal")
            #display_img(contour_img)

            image_info = "mask_"+str(mask_count+1)+"_contour_"+str(countour_count+1)+"_slots_"+str(len(slots))
            fname, extension = os.path.splitext(image_path)
            outfilename = fname+"_"+image_info+extension
            out_path = os.path.join(image_dir,"output", outfilename)
            cv2.imwrite(out_path, contour_img)


    