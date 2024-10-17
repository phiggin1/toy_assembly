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
import pycocotools.mask as mask_util
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct
import json

def reject_outliers(data, m=20):
    d = np.abs(data[:,0] - np.median(data[:,0]))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)

    resp = data[s < m]

    return resp

def create_cloud(points, frame):
    point_list = []
    for p in points:
        x = p[0]
        y = p[1]
        z = p[2]
        pt = [x, y, z, 0]
        r = p[3]
        g = p[4]
        b = p[5]
        a = 255
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        pt[3] = rgb
        point_list.append(pt)

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 16, PointField.UINT32, 1),
    ]

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame
    pc2 = point_cloud2.create_cloud(header, fields, point_list)

    return pc2

def transform_point(ps, mat44, target_frame):
    xyz = tuple(np.dot(mat44, np.array([ps.point.x, ps.point.y, ps.point.z, 1.0])))[:3]
    r = PointStamped()
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

        rospy.loginfo(f"debug:{self.debug}")
        rospy.loginfo(f"server_port: {server_port}")
        rospy.loginfo(f"real world:{self.real}")
        rospy.loginfo(f"arm: {self.arm}")

        if self.real:
            self.cam_link_name = f"{self.arm}_camera_color_frame"
            cam_info_topic = f"/{self.arm}_camera/color/camera_info_throttled"
            rgb_image_topic = f"/{self.arm}_camera/color/image_rect_color_throttled"              
            depth_image_topic = f"/{self.arm}_camera/depth_registered/sw_registered/image_rect_throttled"
            output_image_topic = f"/{self.arm}_camera/color/sam_overlay_raw"
        else:
            self.cam_link_name = f"{self.arm}_camera_link"
            cam_info_topic = f"/unity/camera/{self.arm}/rgb/camera_info"
            rgb_image_topic = f"/unity/camera/{self.arm}/rgb/image_raw"
            depth_image_topic = f"/unity/camera/{self.arm}/depth/image_raw"
            output_image_topic = f"/unity/camera/{self.arm}/rgb/sam_overlay_raw"

        rospy.loginfo(cam_info_topic)
        rospy.loginfo(rgb_image_topic)
        rospy.loginfo(depth_image_topic)

        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.bind("tcp://*:%s" % server_port)
        rospy.loginfo(f"Server listening on port:{server_port}")

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
        
        self.sam_serv = rospy.Service('/get_sam_segmentation', SAM, self.SAM)
        
        self.annote_pub = rospy.Publisher(output_image_topic, Image, queue_size=10)

        rospy.spin()

    def image_cb(self, rgb_ros_image, depth_ros_image):
        success = self.mutex.acquire(block=False, timeout=0.02)
        if success:
            #print(f"rgb eoncoding: {rgb_ros_image.encoding}")
            #print(f"depth eoncoding: {depth_ros_image.encoding}")
            self.rgb_image = self.cvbridge.imgmsg_to_cv2(rgb_ros_image, desired_encoding="bgr8") 
            self.depth_image = self.cvbridge.imgmsg_to_cv2(depth_ros_image)#, desired_encoding="passthrough") 
            if not self.have_images:
                rospy.loginfo("HAVE IMAGES")
                self.have_images = True
            self.mutex.release()

    def SAM(self, request):
        if self.debug: rospy.loginfo('SAM req recv')
        with self.mutex:
            image = self.rgb_image #self.cvbridge.imgmsg_to_cv2(request.image, "bgr8")     
            text = request.text_prompt

            print(text)
            self.colors = {
                "tan tray":(255,0,0),
                "orange tray":(0,255,0),
                "tan horse body":(0,0,255),
                "blue horse legs":(255,255,0),
                "orange horse legs":(255,0,255),
                "table":(255,255,255)
            }

            print(image.shape)

            msg = {"type":"sam",
                "image":image.tolist(),
                "text":text,
            }

            print(msg["text"])

            if self.debug: rospy.loginfo("SAM sending to ada")
            
            self.socket.send_json(msg)
            resp = self.socket.recv_json()
            if self.debug: rospy.loginfo('SAM recv from ada') 

            response = SAMResponse()
            response.annotated_image = self.cvbridge.cv2_to_imgmsg(np.asarray(resp['annotated_image'], dtype=np.uint8), encoding="bgr8")
            response.annotated_image.header.frame_id = self.cam_info.header.frame_id
            response.annotated_image.header.stamp = rospy.Time.now()

            self.annote_pub.publish(response.annotated_image)

            self.points = []
            for detection in resp["annotations"]:

                rospy.loginfo(f"{detection['class_name']}, {detection['score']}, {detection['bbox']}")
                obj = Detection()
                obj.class_name = detection['class_name']
                if isinstance(detection['score'], float):
                    obj.score = detection['score']
                else:
                    obj.score = detection['score'][0]
                obj.rle_encoded_mask = json.dumps(detection['segmentation'])
                
                obj.u_min = int(detection['bbox'][0])
                obj.v_min = int(detection['bbox'][1])

                obj.u_max = int(detection['bbox'][2])
                obj.v_max = int(detection['bbox'][3])

                response.object.append(obj)


            rospy.loginfo('reply')
            return response

    def get_position(self, rle, bbox, class_name):
        #mask = rle2mask(rle, (img_width, img_height)) 
        mask = mask_util.decode([rle])
        u_min = int(bbox[0])
        v_min = int(bbox[1])
        u_max = int(bbox[2])
        v_max = int(bbox[3])

        translation,rotation = self.listener.lookupTransform('world', self.cam_info.header.frame_id, rospy.Time(0))
        mat44 = self.listener.fromTranslationRotation(translation, rotation)

        min_x = 1000.0
        min_y = 1000.0
        min_z = 1000.0
        max_x = -1000.0
        max_y = -1000.0
        max_z = -1000.0

        #get the camera center (cx,cy) and focal length (fx,fy)
        cx = self.cam_model.cx()
        cy = self.cam_model.cy()
        fx = self.cam_model.fx()
        fy = self.cam_model.fy()

        distances = []
        for v in range(v_min, v_max, 2):
            for u in range(u_min, u_max, 2):
                if mask[v][u] > 0:
                    d = (self.depth_image[v][u])
                    #if d > 100 depth is in mm
                    if d > 100:
                        d = d/1000.0
                    if d <= 0.001 or d > 1.0 or np.isnan(d):
                        continue
                    distances.append((d, v, u))

        if len(distances)>0:
            distances = reject_outliers(np.asarray(distances), m=1)
            print(f"{class_name} num points: {len(distances)}")
        else:
            print(print(f"{class_name} no points found"))

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
            
            if class_name in self.colors:
                b = self.colors[class_name][0]
                g = self.colors[class_name][1]
                r = self.colors[class_name][2]
            else:
                b = 255
                g = 255
                r = 255

            self.points.append([t_p.point.x, t_p.point.y, t_p.point.z, r, g, b])

        center = PointStamped()
        center.header.frame_id = 'world'
        center.point.x = (min_x + max_x)/2
        center.point.y = (min_y + max_y)/2
        center.point.z = (min_z + max_z)/2

        print(f"center: x:{((min_x + max_x)/2):.2f}, y:{((min_y + max_y)/2):.2f}, z:{((min_z + max_z)/2):.2f}")        

        return center


if __name__ == '__main__':
    get_target = AdaClient()

