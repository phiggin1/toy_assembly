#!/usr/bin/env python3

import rospy
import roslaunch
import rosgraph
import rostopic

'''
    <node pkg="nodelet" type="nodelet" args="manager" name="record_player_manager" output="screen"/>

    <node name="decompress_rgb_right" pkg="image_transport" type="republish" args="compressed in:=/unity/camera/right/rgb raw out:=/unity/camera/right/rgb/image_raw" />
    <!-- builds pointcloud from the color and depth images -->
    <node pkg="nodelet" type="nodelet" name="cloudify_right" args="load depth_image_proc/point_cloud_xyzrgb record_player_manager ">
        <param name="use_sim_time" value="true" />
        <remap from="depth_registered/image_rect" to="/unity/camera/right/depth/image_raw"/> 
        <remap from="rgb/image_rect_color"  to="/unity/camera/right/rgb/image_raw"/> 
        <remap from="rgb/camera_info" to="/unity/camera/right/rgb/camera_info"/>
        <remap from="depth_registered/points" to="/unity/camera/right/depth/points"/> 
    </node>
  
    <node name="decompress_rgb_left" pkg="image_transport" type="republish" args="compressed in:=/unity/camera/left/rgb raw out:=/unity/camera/left/rgb/image_raw" />
    <!-- builds pointcloud from the color and depth images -->
    <node pkg="nodelet" type="nodelet" name="cloudify_left" args="load depth_image_proc/point_cloud_xyzrgb record_player_manager ">
        <param name="use_sim_time" value="true" />
        <remap from="depth_registered/image_rect" to="/unity/camera/left/depth/image_raw"/> 
        <remap from="rgb/image_rect_color"  to="/unity/camera/left/rgb/image_raw"/> 
        <remap from="rgb/camera_info" to="/unity/camera/left/rgb/camera_info"/>
        <remap from="depth_registered/points" to="/unity/camera/left/depth/points"/> 
    </node>
'''

'''
roslaunch.core.Node(package, node_type, name=None, namespace='/', 
                 machine_name=None, args='', 
                 respawn=False, respawn_delay=0.0, 
                 remap_args=None, env_args=None, output=None, cwd=None, 
                 launch_prefix=None, required=False, filename='<unknown>')

    package: node package name, str
    node_type: node type, str
    name: node name, str
    namespace: namespace for node, str
    machine_name: name of machine to run node on, str
    args: argument string to pass to node executable, str
    respawn: if True, respawn node if it dies, bool
    respawn_delay: if respawn is True, respawn node after delay, float
    remap_args: list of [(from, to)] remapping arguments, [(str, str)]
    env_args: list of [(key, value)] of additional environment vars to set for node, [(str, str)]
    output: where to log output to, either Node, 'screen' or 'log', str
    cwd: current working directory of node, either 'node', 'ROS_HOME'. Default: ROS_HOME, str
    launch_prefix: launch command/arguments to prepend to node executable arguments, str
    required: node is required to stay running (launch fails if node dies), bool
    filename: name of file Node was parsed from, str 
'''

'''

roslaunch.configure_logging(uuid)
launch = roslaunch.scriptapi.ROSLaunch()
launch.parent = roslaunch.parent.ROSLaunchParent(uuid, "path/to/base.launch")
launch.start()

# Start another node
node = roslaunch.core.Node(package, executable)
launch.launch(node)

try:
    launch.spin()
finally:
    # After Ctrl+C, stop all nodes from running
    launch.shutdown()
'''

'''
    <node name="decompress_rgb_right" pkg="image_transport" type="republish" args="compressed in:=/unity/camera/right/rgb raw out:=/unity/camera/right/rgb/image_raw" />
    <!-- builds pointcloud from the color and depth images -->
    <node pkg="nodelet" type="nodelet" name="cloudify_right" args="load depth_image_proc/point_cloud_xyzrgb record_player_manager ">
        <param name="use_sim_time" value="true" />
        <remap from="depth_registered/image_rect" to="/unity/camera/right/depth/image_raw"/> 
        <remap from="rgb/image_rect_color"  to="/unity/camera/right/rgb/image_raw"/> 
        <remap from="rgb/camera_info" to="/unity/camera/right/rgb/camera_info"/>
        <remap from="depth_registered/points" to="/unity/camera/right/depth/points"/> 
    </node>
'''
#roslaunch.configure_logging(uuid)
launch = roslaunch.scriptapi.ROSLaunch()
#launch.parent = roslaunch.parent.ROSLaunchParent(uuid, "path/to/base.launch")
launch.start()
package = "image_transport"
node_type = "republish"
args="compressed in:=/unity/camera/right/rgb raw out:=/unity/camera/right/rgb2/image_raw"
remap = [
        ("depth_registered/image_rect","/unity/camera/right/depth/image_raw"),
        ("rgb/image_rect_color", "/unity/camera/right/rgb/image_raw"),
        ("rgb/camera_info", "/unity/camera/right/rgb/camera_info"),
        ("depth_registered/points", "/unity/camera/right/depth/points")
]
node = roslaunch.core.Node(name="decompress_rgb_right2", package=package, node_type=node_type,args=args)
launch.launch(node)
try:
    launch.spin()
finally:
    # After Ctrl+C, stop all nodes from running
    launch.shutdown()

'''
master = rosgraph.Master('/rostopic')

while not rospy.is_shutdown():
    pubs, subs = rostopic.get_topic_list(master=master)
    topic_data = {}
    print(f"subs {len(subs)}")
    for topic in pubs:
        name = topic[0]
        if name not in topic_data:
            topic_data[name] = {}
            topic_data[name]['type'] = topic[1]
        topic_data[name]['publishers'] = topic[2]

    print(f"subs {len(pubs)}")
    for topic in subs:
        name = topic[0]
        if name not in topic_data:
            topic_data[name] = {}
            topic_data[name]['type'] = topic[1]
        topic_data[name]['subscribers'] = topic[2]

    for topic_name in sorted(topic_data.keys()):
        #print(topic_name)
        if "camera" in topic_name:
            print(topic_name, topic_data[topic_name]["type"])

    print("===============================")
'''