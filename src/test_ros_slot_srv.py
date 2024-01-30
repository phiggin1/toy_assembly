
import rospy
from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel
from geometry_msgs.msg import PointStamped, Point
from toy_assembly.srv import DetectSlot, DetectSlotResponse

rospy.init_node('SlotTracking', anonymous=True)

cam_info_topic =    rospy.get_param("cam_info_topic",  "/unity/camera/rgb/camera_info")
rgb_image_topic =   rospy.get_param("rgb_image_topic",     "/unity/camera/rgb/image_raw")
depth_image_topic = rospy.get_param("depth_image_topic",     "/unity/camera/depth/image_raw")
ocation_topic =    rospy.get_param("location_topic",  "/pt")

rgb_image = rospy.wait_for_message(self.rgb_image_topic, Image) #rospy.Subscriber(self.rgb_image_topic, Image, self.image_cb)
rospy.loginfo("Got RGB image")
depth_image = rospy.wait_for_message(self.depth_image_topic, Image) #rospy.Subscriber(self.rgb_image_topic, Image, self.image_cb)
rospy.loginfo("Got Depth image")
location = rospy.wait_for_message(self.location_topic, PointStamped) #rospy.Subscriber(self.location_topic, PointStamped, self.location_cb)
rospy.loginfo("Got location")
cam_info = rospy.wait_for_message(self.cam_info_topic, CameraInfo)



detect_slot_serv =  rospy.ServiceProxy('get_slot_location', DetectSlot)



resp = detect_slot_serv(rgb_image, depth_image, cam_info, location)

rospy.loginfo(resp)