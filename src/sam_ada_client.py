#!/usr/bin/env python3

import zmq
import tf
import numpy as np
import rospy
import message_filters
from cv_bridge import CvBridge
from toy_assembly.srv import SAM, SAMResponse
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Point
from image_geometry import PinholeCameraModel
from multiprocessing import Lock
from toy_assembly.msg import Detection




def rle2mask(mask_rle, shape, label=1, ):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label

    return img.reshape(shape)  # Needed to align to RLE direction


def transform_point(ps, mat44, target_frame):
    xyz = tuple(np.dot(mat44, np.array([ps.point.x, ps.point.y, ps.point.z, 1.0])))[:3]
    r = PointStamped
    r.header.stamp = ps.header.stamp
    r.header.frame_id = target_frame
    r.point = Point(*xyz)
    return r


class AdaClient:
    def __init__(self):
        rospy.init_node('ada_sam')

        self.mutex = Lock()

        self.listener = tf.TransformListener()
        self.cvbridge = CvBridge()
        
        self.debug = rospy.get_param("~debug", True)
        server_port = rospy.get_param("~port", "8899")

        self.real = rospy.get_param("~real", default=False)
        self.arm = rospy.get_param("~arm", default="left")

        rospy.loginfo(f"real world:{self.real}")
        rospy.loginfo(f"arm: {self.arm}")

        if self.real:
            self.cam_link_name = f"{self.arm}_camera_color_frame"
            cam_info_topic = f"/{self.arm}_camera/color/camera_info_throttled"
            rgb_image_topic = f"/{self.arm}_camera/color/image_rect_color_throttled"              
            depth_image_topic = f"/{self.arm}_camera/depth_registered/sw_registered/image_rect_throttled"
            output_image_topic = f"/{self.arm}_camera/color/overlay_raw"
        else:
            self.cam_link_name = f"{self.arm}_camera_link"
            cam_info_topic = f"/unity/camera/{self.arm}/rgb/camera_info"
            rgb_image_topic = f"/unity/camera/{self.arm}/rgb/image_raw"
            depth_image_topic = f"/unity/camera/{self.arm}/depth/image_raw"
            output_image_topic = f"/unity/camera/{self.arm}/rgb/overlay_raw"

        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % server_port)
        rospy.loginfo(f"Server listening on port:{server_port}")

        self.cam_info = rospy.wait_for_message(cam_info_topic, CameraInfo)
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(self.cam_info)

        self.rgb_sub = message_filters.Subscriber(rgb_image_topic, Image)
        self.depth_sub = message_filters.Subscriber(depth_image_topic, Image)

        ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 10)
        ts.registerCallback(self.image_cb)
        self.rgb_image = None
        self.depth_image = None

        self.sam_serv = rospy.Service('/get_sam_segmentation', SAM, self.SAM)
        
        self.pub = rospy.Publisher(output_image_topic, Image, queue_size=10)

        rospy.spin()

    def image_cb(self, rgb_ros_image, depth_ros_image):
        success = self.mutex.acquire(block=False, timeout=0.02)
        if success:
            self.rgb_image = self.cvbridge.imgmsg_to_cv2(rgb_ros_image) 
            self.depth_image = self.cvbridge.imgmsg_to_cv2(depth_ros_image) 

    def SAM(self, request):
        if self.debug: rospy.loginfo('SAM req recv')

        image = self.cvbridge.imgmsg_to_cv2(request.image, "bgr8")     
        with self.mutex:
            image = self.rgb_image #self.cvbridge.imgmsg_to_cv2(request.image, "bgr8")     
            text = "tan tray. orange tray. tan horse body. blue horse legs. orange horse legs. table. hand."

            print(image.shape)

            msg = {"type":"sam",
                "image":image.tolist(),
                "text":text,
            }

            if self.debug: rospy.loginfo("SAM sending to ada")
            
            with self.mutex:
                self.socket.send_json(msg)
                resp = self.socket.recv_json()
            if self.debug: rospy.loginfo('SAM recv from ada') 

            response = SAMResponse()
            response.annotated_image = self.cvbridge.cv2_to_imgmsg(np.asarray(resp['annotated_image'], dtype=np.uint8), encoding="bgr8")
            response.annotated_image.header.frame_id = request.image.header.frame_id
            response.annotated_image.header.stamp = rospy.time.Now()

            self.pub.publish(response.annotated_image)

            for detection in resp["annotations"]:
                rospy.loginfo(f"{detection['class_name']}, {detection['score']}, {detection['bbox']}")
                obj = Detection()
                obj.class_name = detection['class_name']
                obj.score = detection['score']
                obj.rle_encoded_mask = detection['segmentation']['counts']#self.cvbridge.cv2_to_imgmsg(rle2mask(detection['segmentation']['counts']))
                
                obj.u_min = int(detection['bbox'][0])
                obj.v_min = int(detection['bbox'][1])

                obj.u_max = int(detection['bbox'][2])
                obj.v_max = int(detection['bbox'][3])

                poisition, w, h, d = self.get_position(detection['segmentation']['counts'], detection['bbox'], self.cam_info.width, self.cam_info.height)
                obj.center = poisition
                obj.width = w
                obj.height = h
                obj.depth = d
                response.object.append(obj)

            
            return response

    def get_position(self, rle, bbox, img_width, img_height):
        mask = rle2mask(rle) 
        u_min = bbox[0]
        v_min = bbox[1]
        u_max = bbox[2]
        v_max = bbox[3]

        target_frame = 'world'
        translation,rotation = self.listener.lookupTransform(target_frame, self.cam_model.header.frame_id, rospy.Time(0))
        mat44 = self.listener.fromTranslationRotation(translation, rotation)

        min_x = 1000.0
        min_y = 1000.0
        min_z = 1000.0
        max_x = -1000.0
        max_y = -1000.0
        max_z = -1000.0
        for u in range(u_min, u_max):
            for v in range(v_min, v_max):
                if mask[u][v] > 0:
                    print(self.depth_image[u][v])
                    print(self.cam_model.projectPixelTo3dRay(self.depth_image[u][v]))
                    ps = PointStamped()
                    ps.header.frame_id = self.cam_model.header.frame_id
                    
                    t_p = transform_point(ps, mat44, target_frame)
                    if t_p.point.x > max_x:
                        max_x = t_p.point.x
                    if t_p.point.x < min_x:
                        min_x = t_p.point.x

                    if t_p.point.y > max_y:
                        max_y = t_p.point.y
                    if t_p.point.y < min_y:
                        min_y = t_p.point.y

                    if t_p.point.z > max_z:
                        max_z = t_p.point.z
                    if t_p.point.z < min_z:
                        min_z = t_p.point.z

        center = Point()
        center.x = (min_x + max_x)/2
        center.y = (min_y + max_y)/2
        center.z = (min_z + max_z)/2
        w = max_y - min_y
        h = max_z - min_z
        d = max_x - min_x

        return center, w, h, d


if __name__ == '__main__':
    get_target = AdaClient()

