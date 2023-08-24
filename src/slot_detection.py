import numpy as np
import cv2
import matplotlib.pyplot as plt


def display_img(img):
    # show image
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




angle_threshold = 0.2
pixel_distance_threshold = 15
eps  = 3.5 #works well with real images


#target_x = 460
#target_y = 375
#image_path = 'test.jpg'
#image_path = 'real_test_image.jpg'


target_x = 900
target_y = 375
image_path = 'real_body_test.jpg'


img = cv2.imread(image_path)
print(img.shape)
#display_img(img)
desired_image_height = 960
desired_image_width = 1440
old_image_height, old_image_width, channels = img.shape
bottom_padding = desired_image_height-old_image_height
right_padding = desired_image_width-old_image_width
img = cv2.copyMakeBorder(img,0,bottom_padding,0,right_padding,cv2.BORDER_CONSTANT,value=[0,0,0])
print(img.shape)


from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["default"](checkpoint="../../sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
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

    #get the contours
    contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    print("num contours: "+str(len(contours)))

    #if there is more than one contour the segmentation did something wierd 
    # so skip the mask
    if len(contours)>1:
         continue
    
    for (countour_count,contour) in enumerate(contours):
        contour_img = img.copy()
        #display target on image
        cv2.circle(contour_img, (target_x, target_y), radius=3, color=(0, 0, 255), thickness=-1)

        print("contour: "+str(countour_count+1)+" of "+str(len(contours)))
        #display the contours overlayed on copy of origional image
        cv2.drawContours(contour_img, contours, -1, (0,255,0), 1)

        #  eps = 1.0
        contour2 = cv2.approxPolyDP(contour, eps, closed=True)
        print("   orig contour len: "+str(len(contour)))
        print("reduced contour len: "+str(len(contour2)))
        contour=contour2

        right_angles = []
        slots = []
        num_points = len(contour)
        print("num points: "+str(num_points))

        for i in range(num_points-3):
            cv2.circle(contour_img, (contour[i][0][0], contour[i][0][1]), radius=2, color=(0, 0, 255), thickness=-1)
            #for the next four points get the 3 vectors between them (0 to 1, 1 to 2, 2 to 3) 
            v1 = contour[i+1]-contour[i]
            v2 = contour[i+2]-contour[i+1]
            v3 = contour[i+3]-contour[i+2]

            #calculate the angle between each sequential pair of vectors
            cos_theta1 = np.dot(v1[0],v2[0])/(np.linalg.norm(v1[0])*np.linalg.norm(v2[0]))
            cos_theta2 = np.dot(v2[0],v3[0])/(np.linalg.norm(v2[0])*np.linalg.norm(v3[0]))

            #check for 4 sequential points that are at ~90deg apart
            #and not too close together
            if abs(cos_theta1) < angle_threshold and abs(cos_theta2) < angle_threshold:
                    if np.linalg.norm(v1[0]) > pixel_distance_threshold and np.linalg.norm(v2[0]) > pixel_distance_threshold and np.linalg.norm(v3[0]) > pixel_distance_threshold:
                        right_angles.append(contour[i])
                        right_angles.append(contour[i+1])
                        right_angles.append(contour[i+2])
                        right_angles.append(contour[i+3])
                        slots.append( (contour[i], contour[i+1], contour[i+2], contour[i+3]) )

        #print(len(right_angles))
        #for i in range(len(right_angles)):
        #    print(i, right_angles[i][0])
        print("num slots: "+str(len(slots)))
        print(slots)
        #                                                 B       G   R
        cv2.drawContours(contour_img, right_angles,  -1, (255,    0,  0), 8)
        print("final")
        display_img(contour_img)
    
    