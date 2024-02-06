#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from image_geometry import PinholeCameraModel
from obj_segmentation.msg import SegmentedClustersArray
from toy_assembly.srv import SAM
from toy_assembly.srv import CLIP

BLUE = (255, 0, 0)
GREEN = (0,255,0)
RED = (0,0,255)
PURPLE = (255,0,128)

def get_centroid(pc):
    min_x = 1000.0
    min_y = 1000.0
    min_z = 1000.0
    max_x = -1000.0
    max_y = -1000.0
    max_z = -1000.0

    #for each object get a bounding box
    for p in pc2.read_points(pc):
        if p[0] > max_x:
            max_x = p[0]
        if p[0] < min_x:
            min_x = p[0]

        if p[1] > max_y:
            max_y = p[1]
        if p[1] < min_y:
            min_y = p[1]

        if p[2] > max_z:
            max_z = p[2]
        if p[2] < min_z:
            min_z = p[2]

    center = [(min_x + max_x)/2, (min_y + max_y)/2, (min_z + max_z)/2]
    w = max_x-min_x
    h = max_y-min_y
    d = max_z-min_z

    return center

def display_img(img):
    # show image
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

rospy.init_node('test_clip')
cvbridge = CvBridge()


clip_serv = rospy.ServiceProxy('get_clip_probabilities', CLIP)
segment_serv = rospy.ServiceProxy('get_sam_segmentation', SAM)

max_num_contours = 12

rgb_image_topic =  "/unity/camera/right/rgb/image_raw"
cam_info_topic =   "/unity/camera/right/rgb/camera_info"
cluster_topic =    "/object_clusters"
transcript_topic = "/transcript"

rospy.loginfo(f"rgb_image_topic:{rgb_image_topic}")
rospy.loginfo(f"cam_info_topic:{cam_info_topic}")
rospy.loginfo(f"cluster_topic:{cluster_topic}")
rospy.loginfo(f"transcript_topic:{transcript_topic}")
            
rgb_image = rospy.wait_for_message(rgb_image_topic, Image) 
rospy.loginfo("Got RGB image")

cam_info = rospy.wait_for_message(cam_info_topic, CameraInfo)
rospy.loginfo("Got cam_info")
cam_model = PinholeCameraModel()
cam_model.fromCameraInfo(cam_info)

clusters = rospy.wait_for_message(cluster_topic, SegmentedClustersArray)
rospy.loginfo("Got clusters")


#transcript = [rospy.wait_for_message(transcript_topic, String)]
transcript = ["red horse", "blue horse"]
rospy.loginfo("Got transcript") 
rospy.loginfo(transcript) 


images = []
positions = []

for i, pc in enumerate(clusters.clusters):
    print(f"obj {i}")
    p = get_centroid(pc)
    positions.append(p)

    u, v = cam_model.project3dToPixel( p )
    print(p)
    print(u,v)

    target_x = int(u)
    target_y = int(v)
    resp  = segment_serv(rgb_image, target_x, target_y)
    #scores = resp.scores
    rgb_cv = cvbridge.imgmsg_to_cv2(rgb_image)

    #display target on image
    disp_img = rgb_cv.copy()
    cv2.circle(disp_img, (target_x, target_y), radius=5, color=PURPLE, thickness=-1)
    #display_img(disp_img)

    for j, mask in enumerate(resp.masks):
        rospy.loginfo(f"mask {j} of {len(resp.masks)}")
        #get image
        mask_cv = cvbridge.imgmsg_to_cv2(mask)

        '''
        rospy.loginfo(f"shape:{mask_cv.shape}")
        rospy.loginfo(f"min:{np.min(mask_cv)}")
        rospy.loginfo(f"max:{np.max(mask_cv)}")
        '''

        imgray = np.asarray(mask_cv, dtype=np.uint8)
        #display_img(imgray)

        contours, _ = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        if len(contours) > max_num_contours:
            rospy.loginfo(f"{len(contours)}>{max_num_contours}")
            continue

        '''
        contour_img = imgray.copy()
        contour_img = cv2.cvtColor(contour_img,cv2.COLOR_GRAY2RGB)
        #display target on image
        cv2.circle(contour_img, (target_x, target_y),  radius=3, color=RED, thickness=-1)
        #display the contours overlayed on copy of origional image
        cv2.drawContours(contour_img, contours, -1, green, 1)
        display_img(contour_img)
        '''

        masked_image = cv2.bitwise_and(rgb_cv, rgb_cv, mask=mask_cv.astype(np.uint8))
        #display_img(masked_image)

        images.append(cvbridge.cv2_to_imgmsg(masked_image, "bgr8"))
        #TODO get scores from SAM and use along with #contours to filter masks better
        rospy.loginfo("Have decent image (TODO get scores from SAM and use along with #contours to filter masks better)")
        break

rospy.loginfo(f"#images:{len(images)}")
rospy.loginfo(f"#transcripts:{len([transcript])}")

clip_probs = clip_serv(images, transcript)
dims = tuple(map(lambda x: x.size, clip_probs.probs.layout.dim))
probs = np.array(clip_probs.probs.data, dtype=np.float32).reshape(dims).astype(dtype=np.float32)

rospy.loginfo(transcript)
rospy.loginfo(positions)
rospy.loginfo(probs)