#!/usr/bin/env python3

import sys
import rospy
import numpy as np
import math
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import Marker, MarkerArray

from image_geometry import PinholeCameraModel
import sensor_msgs.point_cloud2 as pc2
from toy_assembly.srv import SAM
from toy_assembly.srv import CLIP
from toy_assembly.srv import MoveITGrabPose, MoveITPose, MoveITPoseResponse
from toy_assembly.srv import OrientCamera
from std_srvs.srv import Trigger 

import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose
from geometry_msgs.msg import TwistStamped, PoseStamped, PointStamped
from geometry_msgs.msg import Vector3
import tf
from tf.transformations import quaternion_from_euler
import json

class Right_arm:
    def __init__(self):
        rospy.init_node("rightArm")
        #org right starting pose
        #self.start_pose = [0.0, 0.0, -1.0, 0.0, -2.0, 1.57]

        self.real = rospy.get_param("~real", True)

        self.arm = "right"
        self.other_arm = "left"
        self.base_frame = "world"
        self.padding = 0.075
        left_frames = [
            "left_base_link",
            "left_shoulder_link",
            "left_bicep_link",
            "left_forearm_link",
            "left_spherical_wrist_1_link",
            "left_spherical_wrist_2_link",
            "left_bracelet_link",
            "left_end_effector_link",
            "left_tool_frame"
        ]
        right_frames = [
            "right_base_link",
            "right_shoulder_link",
            "right_bicep_link",
            "right_forearm_link",
            "right_spherical_wrist_1_link",
            "right_spherical_wrist_2_link",
            "right_bracelet_link",
            "right_end_effector_link",
            "right_tool_frame"
        ]
        self.frames = {
            "left": left_frames,
            "right": right_frames
        }

        #self.start_pose = [0.0, 0.0, -1.57, 0.0, -1.57, 0.0]
        self.start_pose = [0.0, 0.33, -1.96, 0.0, -0.90, 0.0]
        self.init_position()	

        self.finger_open = 0.01
        if self.real:
            self.finger_closed = 0.78
        else:
            self.finger_closed = 0.75

        self.hand_closed = [self.finger_closed, self.finger_closed, self.finger_closed, self.finger_closed, self.finger_closed, self.finger_closed]
        self.hand_open = [self.finger_open, self.finger_open, self.finger_open, self.finger_open, self.finger_open, self.finger_open]

        self.gripper_group_name = "gripper"
        self.gripper_move_group = moveit_commander.MoveGroupCommander(self.gripper_group_name)
        self.gripper_move_group.set_max_velocity_scaling_factor(1.0)

        self.horse_topic = '/object_positions'
        self.horse_pose = None  
        #self.horses = rospy.Subscriber(self.horse_topic, PoseStamped, self.get_horse_pose)
        #self.horse_pose = []
        self.grabbed_object = False

        self.gripper_orientation = {'hand_pointing_right_cam_up': [math.sqrt(2)/2, 0, 0, math.sqrt(2)/2],
                            'hand_pointing_right_cam_front': [0.5, 0.5, -0.5, 0.5],
                            'hand_pointing_left_cam_up': [0, math.sqrt(2)/2, math.sqrt(2)/2, 0],
                            'hand_pointing_left_cam_front': [-0.5, -0.5, -0.5, -0.5],
                            'hand_pointing_down_cam_right': [-1, 0, 0, 0],
                            'hand_pointing_down_cam_front': [-math.sqrt(2)/2, -math.sqrt(2)/2, 0, 0],
                            'hand_pointing_forward_cam_up': [0.5, 0.5, 0.5, 0.5],
                            'hand_pointing_forward_cam_right' : [math.sqrt(2)/2, 0, math.sqrt(2)/2, 0]}

        self.listener = tf.TransformListener()
        self.grab = rospy.Publisher('/buttons', String, queue_size=10)
        #self.release = rospy.Publisher('/buttons', String, queue_size=10)

        self.move_pose = rospy.Service('move_pose', MoveITPose, self.move_to_pose)

        self.grab_object = rospy.Service('grab_object', MoveITGrabPose, self.get_object)
        self.release_object = rospy.Service('release_object', MoveITGrabPose, self.place_object)

        self.open_hand = rospy.Service('open_hand', Trigger, self.open_gipper)
        self.close_hand = rospy.Service('close_hand', Trigger, self.close_gipper)

        self.get_orientation = rospy.Service('get_orientation', MoveITGrabPose, self.find_orientation)

        self.obj_marker_pub = rospy.Publisher("obj_marker", Marker, queue_size=10)
        self.obj_marker = Marker()
        self.obj_marker.header.frame_id = 'world'
        self.obj_marker.type = Marker.CUBE
        self.obj_marker.color.r = 0.0
        self.obj_marker.color.g = 1.0
        self.obj_marker.color.b = 1.0
        self.obj_marker.color.a = 1.0

        self.rotate_object = rospy.Service('rotate_object', OrientCamera, self.change_orientation)
        
        self.min_right = Vector3()
        self.max_right = Vector3()
        self.min_left = Vector3()
        self.max_left = Vector3()
        self.marker_pub = rospy.Publisher(f"{self.arm}_arm_bbox", Marker, queue_size=10)
        self.marker_sub = rospy.Subscriber(f"{self.other_arm}_arm_bbox", Marker, self.get_other_bbox)
        self.twist_sub = rospy.Subscriber("/my_gen3_right/workspace/delta_twist_cmds", TwistStamped, self.twist_cb)
        self.twist_pub = rospy.Publisher("/my_gen3_right/servo/delta_twist_cmds", TwistStamped, queue_size=10)
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            marker = self.get_marker(self.arm)
            self.marker_pub.publish(marker)
            rate.sleep()

        #rospy.spin()
        
    def twist_cb(self, twist):
        #rospy.loginfo(twist)

        self.get_self_bbox(self.get_marker(self.arm))

        #have the bounding boxes for both arms
        #print(f"right: {self.min_right.x:,.4f}, {self.max_right.x:,.4f}, {self.min_right.y:,.4f}, {self.max_right.y:,.4f}, {self.min_right.z:,.4f}, {self.max_right.z:,.4f}")
        #print(f"left: {self.min_left.x:,.4f}, {self.max_left.x:,.4f}, {self.min_left.y:,.4f}, {self.max_left.y:,.4f}, {self.min_left.z:,.4f}, {self.max_left.z:,.4f}")
        ee_position = self.arm_move_group.get_current_pose().pose.position
        


        l_x = twist.twist.linear.x
        l_y = twist.twist.linear.y
        l_z = twist.twist.linear.z

        a_x = twist.twist.angular.x
        a_y = twist.twist.angular.y
        a_z = twist.twist.angular.z
        
        #rospy.loginfo(f"ee pos.z: {ee_position.z} l_z:{l_z}")

        if self.check_collision() and l_y > 0:
            rospy.loginfo(f"arms collision: {l_y}")
            l_y = 0.0

        if ee_position.x > 0.6 and l_x > 0:
            rospy.loginfo(f"too far forward, {l_x}, {ee_position.x}")
            l_x = 0.0

        if ee_position.z < 0.05 and l_z < 0:
            rospy.loginfo(f"too low, {l_z}, {ee_position.z}")
            l_z = 0.0
        elif ee_position.z < 0.15 and l_z < 0:
            l_z = 0.3*l_z

            l_x = 0.3*l_x
            l_y = 0.3*l_y
            rospy.loginfo(f"nearing table slowing down low, {l_z}, {ee_position.z}")
        elif ee_position.z < 0.3 and l_z < 0:
            l_z = 0.6*l_z

            l_x = 0.6*l_x
            l_y = 0.6*l_y
            rospy.loginfo(f"getting low slowing down, {l_z}, {ee_position.z}")


        #rospy.loginfo(f"after ee pos.z: {ee_position.z} l_z:{l_z}")

        new_twist = TwistStamped()
        new_twist.header.frame_id = twist.header.frame_id
        new_twist.header.seq = twist.header.seq
        new_twist.header.stamp = rospy.Time.now()
        new_twist.twist.linear.x = l_x
        new_twist.twist.linear.y = l_y
        new_twist.twist.linear.z = l_z
        new_twist.twist.angular.x = a_x
        new_twist.twist.angular.y = a_y
        new_twist.twist.angular.z = a_z

        '''
        rospy.loginfo(f"right arm controller \n twist: {twist.header.frame_id}")
        rospy.loginfo(f"right arm controller \n new_twist: {new_twist.header.frame_id}")
        rospy.loginfo(f"right arm controller \n twist: {twist.twist}")
        rospy.loginfo(f"right arm controller \n new_twist: {new_twist.twist}")
        '''

        self.twist_pub.publish(new_twist)


    def init_position(self):
        moveit_commander.roscpp_initialize(sys.argv)
        #rospy.init_node("move_to_start", anonymous=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_group_name = "arm"
        self.arm_move_group = moveit_commander.MoveGroupCommander(self.arm_group_name)
        

        self.planning_frame = self.arm_move_group.get_planning_frame()
        print(f"planning frame:{self.planning_frame}")

        rospy.sleep(2)
        
        table_pose = PoseStamped()
        table_pose.header.frame_id = "world"
        table_pose.pose.position.x = 0.0
        table_pose.pose.position.y = 0.4
        table_pose.pose.position.z = -0.5
        table_pose.pose.orientation.w = 1.0
        self.scene.add_box("table", table_pose, size=(2.0, 3.0, 1.0))
        print("table", self.wait_for_scene_update("table", 4))
        
        person_pose = PoseStamped()
        person_pose.header.frame_id = "world"
        person_pose.pose.position.x = 1.0
        person_pose.pose.position.y = 0.4
        person_pose.pose.position.z = 0.0
        person_pose.pose.orientation.w = 1.0
        self.scene.add_box("person", person_pose, size=(0.5, 3.0, 2.0))
        print("person", self.wait_for_scene_update("person", 4))
        
        other_arm_pose = PoseStamped()
        other_arm_pose.header.frame_id = "world"
        other_arm_pose.pose.position.x = 0.0
        other_arm_pose.pose.position.y = 0.4
        other_arm_pose.pose.position.z = 0.4125
        other_arm_pose.pose.orientation.w = 1.0
        self.scene.add_box("other_arm", other_arm_pose, size=(0.85, 0.85, 0.85))
        print("other_arm", self.wait_for_scene_update("other_arm", 4))

        self.gripper_group_name = "arm"
        self.gripper_move_group = moveit_commander.MoveGroupCommander(self.gripper_group_name)

        self.arm_move_group.set_max_velocity_scaling_factor(0.750)
        #self.arm_move_group.set_goal_position_tolerance(self.goal_tolerance)
        self.arm_move_group.go(self.start_pose, wait=True) 
        self.arm_move_group.stop()

    def wait_for_scene_update(self, name, timeout):
        start = rospy.get_time()

        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
                is_known = name in self.scene.get_known_object_names()
                if is_known:
                    return True
                rospy.sleep(0.1)
                seconds = rospy.get_time()

        return False
    
    def get_horse_pose(self, pose):
        """
        gets the 3d location of objects in space specified in the cluster
        """
        #del self.horse_pose[:]
        print(f'pose: {pose}')
        self.horse_pose = self.transform_obj_pos(pose)
        print(f'transformed: {self.horse_pose}')

    def move_to_pose(self, request):
        rospy.loginfo("---=== move_to_pose ===---")
        object_pose = self.transform_obj_pos(request.pose)        
        rospy.loginfo(object_pose)

        pose_goal = Pose()
        pose_goal.position = object_pose.pose.position
        pose_goal.orientation = object_pose.pose.orientation

        print(pose_goal)

        status = False
        self.arm_move_group.set_pose_target(pose_goal)
        status = self.arm_move_group.go(pose_goal, wait = True)
        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()

        resp = MoveITPoseResponse()
        resp.result = status
        rospy.loginfo(resp)

        return resp

    def get_object(self, request):
        status = False
        #object_pose = self.transform_obj_pos(request.pose)        
        position, orientation, height = self.find_orientation(request)
        print(f"obj height: {height}m")
        #if the height of the object is under 2.5cm
        # the object is lying flat on the table/tray
        # or being held in a way to make grabbing 'impossible'
        if height < 0.025:
            return status
        #print(object_pose)
        pose_goal = Pose()
        pose_goal.position.x = position.x
        pose_goal.position.y = position.y
        pose_goal.position.z = position.z + 0.1
        pose_goal.orientation.x = orientation.x
        pose_goal.orientation.y = orientation.y
        pose_goal.orientation.z = orientation.z
        pose_goal.orientation.w = orientation.w

        print(f"stand off: {pose_goal}")

        #print('timeout 2')
        self.arm_move_group.set_pose_target(pose_goal)

        #print('timeout3')
        print(pose_goal)
        self.arm_move_group.go(pose_goal, wait=True)
        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()
        
        #print(object_pose)
        pose_goal2 = Pose()
        pose_goal2.position.x = position.x
        pose_goal2.position.y = position.y
        pose_goal2.position.z = position.z
        pose_goal2.orientation.x = orientation.x
        pose_goal2.orientation.y = orientation.y
        pose_goal2.orientation.z = orientation.z
        pose_goal2.orientation.w = orientation.w
        #print('timeout1')

        print(f"end pose: {pose_goal2}")

        
        status = self.arm_move_group.go(pose_goal2, wait = True)
        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()
        

        """
        publishing "grabbed" vs. "released" will do as follows in unity with the nearest object
        """

        # close fingers
        self.close_gipper(Trigger())

        self.arm_move_group.set_max_velocity_scaling_factor(0.750)
        self.arm_move_group.go(pose_goal, wait=True) 
        self.arm_move_group.stop()

        self.grabbed_object = status
        return status
        
    def close_gipper(self, req):
        """
        publishing "grabbed" vs. "released" will do as follows in unity with the nearest object
        """

        # close fingers
        a = dict()
        a["robot"] = "right"
        a["action"] = "grab"
        s = json.dumps(a)
        self.grab.publish(s)

        self.gripper_move_group.set_max_velocity_scaling_factor(0.1)
        status = self.gripper_move_group.go(self.hand_closed, wait=True) 
        self.gripper_move_group.stop()
        self.gripper_move_group.set_max_velocity_scaling_factor(1.0)

        resp = Trigger._response_class()
        resp.success = status

        return resp
        
    def open_gipper(self, req):
        """
        publishing "grabbed" vs. "released" will do as follows in unity with the nearest object
        """

        # close fingers
        a = dict()
        a["robot"] = "right"
        a["action"] = "release"
        s = json.dumps(a)
        self.grab.publish(s)
        
        status = self.gripper_move_group.go(self.hand_open, wait=True) 
        self.gripper_move_group.stop()

        resp = Trigger._response_class()
        resp.success = status

        return resp

    def transform_obj_pos(self, obj_pos):
        t = rospy.Time.now()
        obj_pos.header.stamp = t
        #print(f"org frame:{obj_pos.header.frame_id} new frame:{self.planning_frame}")
        self.listener.waitForTransform(obj_pos.header.frame_id, self.planning_frame, t, rospy.Duration(4.0))
        #print("pre", obj_pos)
        obj_pos = self.listener.transformPose(self.planning_frame, obj_pos)
        #print("post", obj_pos)
        return obj_pos

    def place_object(self, request):
        if (self.grabbed_object == True):
            human_hand = self.transform_obj_pos(request.pose)
            #print(object_pose)
            pose_goal = Pose()
            #print('timeout1')
            pose_goal.position = human_hand.pose.position
            quat = quaternion_from_euler(math.pi, 0.0, 0.0)
            #print(quat)
            pose_goal.orientation.x = quat[0]
            pose_goal.orientation.y = quat[1]
            pose_goal.orientation.z = quat[2]
            pose_goal.orientation.w = quat[3]
               
            self.arm_move_group.set_pose_target(pose_goal)

            status = False
            status = self.arm_move_group.go(pose_goal2, wait = True)
            self.arm_move_group.stop()
            self.arm_move_group.clear_pose_targets()

            # open fingers
            self.gripper_move_group.go(self.hand_open, wait=True) 
            self.gripper_move_group.stop()

            a = dict()
            a["robot"] = "right"
            a["action"] = "release"
            s = json.dumps(a)
            self.grab.publish(s)
            self.grabbed_object = status


            return status
        else:
            return False    

    def change_orientation(self, request):
        print(f"change_orientation to {request}")
        pose_goal = Pose()    
        pose_goal.position = self.arm_move_group.get_current_pose()
        orientationList = []

        # take request string and get corresponding orientation values
        if request in self.gripper_orientation.keys():
            orientationList = self.gripper_orientation.get(request)
        else:
            print('Invalid orientation request')
            orientationList = self.arm_move_group.get_current_pose().orientation

        return orientationList

    def get_marker(self, arm):
        points = []
        target = PointStamped()
        for frame in self.frames[arm]:
            self.listener.waitForTransform(frame, self.base_frame, rospy.Time(), rospy.Duration(4.0) )
            target.header.frame_id = frame
            target.header.stamp = rospy.Time()
            transformned_target = self.listener.transformPoint(self.base_frame, target)
            points.append((transformned_target.point,frame))

        marker = self.get_bbox(points)
        
        return marker
        
    def get_bbox(self, points):
        min_x =  99.9
        max_x = -99.9
        min_y =  99.9
        max_y = -99.9
        min_z =  99.9
        max_z = -99.9
        for p, frame in points:
            #print(f"{frame}: {p.x}, {p.y}, {p.z}")
            if p.x > max_x:
                max_x = p.x
            if p.x < min_x:
                min_x = p.x

            if p.y > max_y:
                max_y = p.y
            if p.y < min_y:
                min_y = p.y

            if p.z > max_z:
                max_z = p.z
            if p.z < min_z:
                min_z = p.z

        min_x -= self.padding
        max_x += self.padding
        min_y -= self.padding
        max_y += self.padding
        min_z -= self.padding
        max_z += self.padding

        #print(f"{self.arm}: {min_x:,.4f}, {max_x:,.4f}, {min_y:,.4f}, {max_y:,.4f}, {min_z:,.4f}, {max_z:,.4f}")

        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.type = Marker.CUBE
        marker.pose.position.x = (max_x + min_x)/2.0
        marker.pose.position.y = (max_y + min_y)/2.0
        marker.pose.position.z = (max_z + min_z)/2.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = (max_x - min_x)
        marker.scale.y = (max_y - min_y)
        marker.scale.z = (max_z - min_z)

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.25

        return marker

    def get_self_bbox(self, marker):
        x = marker.pose.position.x# = (max_x + min_x)/2.0
        y = marker.pose.position.y# = (max_y + min_y)/2.0
        z = marker.pose.position.z# = (max_z + min_z)/2.0
        depth  = marker.scale.x# = (max_x - min_x)
        width  = marker.scale.y# = (max_y - min_y)
        height = marker.scale.z# = (max_z - min_z)

        min_x = (2*x - depth)/2.0
        max_x = min_x + depth

        min_y = (2*y - width)/2.0
        max_y = min_y + width

        min_z = (2*z - height)/2.0
        max_z = min_z + height

        self.min_right.x = min_x
        self.min_right.y = min_y
        self.min_right.z = min_z
        self.max_right.x = max_x
        self.max_right.y = max_y
        self.max_right.z = max_z
        self.have_right = True
    
    def get_other_bbox(self, marker):
        x = marker.pose.position.x# = (max_x + min_x)/2.0
        y = marker.pose.position.y# = (max_y + min_y)/2.0
        z = marker.pose.position.z# = (max_z + min_z)/2.0
        depth  = marker.scale.x# = (max_x - min_x)
        width  = marker.scale.y# = (max_y - min_y)
        height = marker.scale.z# = (max_z - min_z)

        min_x = (2*x - depth)/2.0
        max_x = min_x + depth

        min_y = (2*y - width)/2.0
        max_y = min_y + width

        min_z = (2*z - height)/2.0
        max_z = min_z + height

        self.min_left.x = min_x
        self.min_left.y = min_y
        self.min_left.z = min_z
        self.max_left.x = max_x
        self.max_left.y = max_y
        self.max_left.z = max_z
        self.have_left = True
    
    def check_collision(self):
        #print(f"left  : {self.min_left.x:,.4f}, {self.max_left.x:,.4f}, {self.min_left.y:,.4f}, {self.max_left.y:,.4f}, {self.min_left.z:,.4f}, {self.max_left.z:,.4f}")
        #print(f"right : {self.min_right.x:,.4f}, {self.max_right.x:,.4f}, {self.min_right.y:,.4f}, {self.max_right.y:,.4f}, {self.min_right.z:,.4f}, {self.max_right.z:,.4f}")
        if self.min_left.y < self.max_right.y:
            rospy.loginfo(f"possible collsion; {self.min_left.y - self.max_right.y}")

        return (self.min_left.y < self.max_right.y)
    
    def find_orientation(self, msg):
        threshold = 0.6   #change val
        position, width, height, depth = self.get_width(msg)
        ratio = width/depth
        print(f"ratio: {ratio}")

        if ratio > threshold:
            object_orientation = 'orthogonal'
            orientationList = self.change_orientation('hand_pointing_down_cam_right')
            orientation = Quaternion()
            orientation.x = orientationList[0]
            orientation.y = orientationList[1]
            orientation.z = orientationList[2]
            orientation.w = orientationList[3]
            return position, orientation, height#self.change_orientation('hand_pointing_down_cam_right')
        else:
            object_orientation = 'parallel'
            orientationList = self.change_orientation('hand_pointing_down_cam_front')
            orientation = Quaternion()
            orientation.x = orientationList[0]
            orientation.y = orientationList[1]
            orientation.z = orientationList[2]
            orientation.w = orientationList[3]
            return position, orientation, height#self.change_orientation('hand_pointing_down_cam_front')
        
    def get_width(self, msg):  
        #points = pc2.read_points(msg.cloud, field_names=("x", "y", "z"), skip_nans=True)
        
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        min_z = float('inf')
        max_z = float('-inf')
        
        for point in pc2.read_points(msg.cloud, field_names=("x", "y", "z"), skip_nans=True):
            x = point[0]
            y = point[1]
            z = point[2] 
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            if z < min_z:
                min_z = z
            if z > max_z:
                max_z = z
        
        depth = max_x - min_x
        width = max_y - min_y
        height = max_z - min_z

        self.obj_marker.pose.position.x = (max_x + min_x) / 2
        self.obj_marker.pose.position.y = (max_y + min_y) / 2
        self.obj_marker.pose.position.z = (max_z + min_z) / 2

        self.obj_marker.scale.x = depth
        self.obj_marker.scale.y = width
        self.obj_marker.scale.z = height

        self.obj_marker_pub.publish(self.obj_marker)

        return self.obj_marker.pose.position, width, height, depth

if __name__ == '__main__':
    right_robot = Right_arm()
