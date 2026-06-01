#!/usr/bin/env python3
# -*- coding=UTF-8 -*-
"""
Copyright (c) 2024 [Ruiman Intelligent Technology Co., Ltd.]. All rights reserved.
Author: Robert Time: 2024/07/20

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributed source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributed code in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

This software is provided by the copyright holders and contributors "as is" without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose.

In no event shall the copyright holders or contributors be liable for any direct, indirect, incidental, special, punitive, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) even if advised of the possibility of such damage,

whether in contract, strict liability, or tort (including negligence or otherwise) even if advised of the possibility of such damage.

This module allows the callback functions of each topic subscriber of the AGV chassis to be called directly, which facilitates the development of corresponding functions of subsequent ROS nodes.

This module provides [
single target preset point navigation, single target point navigation, multi-target preset point navigation, navigation cancellation, obtaining robot global status,
correcting robot posture, obtaining AGV chassis power, speed control, various maximum speed settings, obtaining parameter values, chassis RGB three-color light settings
] functions. These functions are all encapsulated in the form of functions and wait for ROS topic subscribers to call. It can be understood as a bridge between ROS and the chassis, playing the role of a driver.

Example usage:
rospy.init_node('process', anonymous=True)
rospy.Subscriber("/navigation_marker", String,
callback_navigation_marker)
Other function functions are used in a similar way.
"""

import socket

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float64

from agv_ros.msg import NavigationLocation  
from agv_ros.msg import NavigationJoyControl
from agv_ros.msg import NavigationLedSetColor


def callback_navigation_marker(data: String):
	
	"""
	Callback function of topic/navigation_marker

	Subscribe to the String type data of topic/navigation_marker and send the result to AGV to achieve single-point navigation

	Args:
	data (String): String data type under std_msgs.msg in ROS, data monitored by subscribing to the topic
	"""
	print("\n \n RECIEVED", data.data)
	rospy.loginfo(f"callback_navigation_marker Received letter: {data.data}")
	target_marker = data.data
	move_command = f'/api/move?marker={target_marker}'	# /api/move?marker={dianA} , move to the target point coded "dianA"
	chassis_client.send(move_command.encode('utf-8'))
	rospy.loginfo(f"send chassis_client :/api/move?marker={target_marker}&max_continuous_retries=1")
        
def callback_navigation_location(data: NavigationLocation):
	
	"""
	Callback function of topic/navigation_location
	Subscribe to the NavigationLocation type data of topic/navigation_location and send the result to AGV to navigate to the specified coordinate location

	Args:
	data (NavigationLocation): NavigationLocation data type under agv.msg in ROS, data listened to by subscribing to the topic
	"""
	
	rospy.loginfo(f"callback_navigation_location Received letter: {data}")
	target_x_y_theta = f'{data.x},{data.y},{data.theta}'	# /api/move?location=15.0,4.0,1.5707963
	move_command = f'/api/move?location={target_x_y_theta}'	# Move to the target point at location (15.0, 4.0, pai/2)
	chassis_client.send(move_command.encode('utf-8'))
	rospy.loginfo(f"send chassis_client :/api/move?location={target_x_y_theta}&max_continuous_retries=1")   
	full_response = ''
	
def callback_navigation_multipoint(data: String):
	
	"""
	Callback function of topic/navigation_multipoint

	Subscribe to the String type data of topic/navigation_multipoint and send the result to AGV to realize the multi-target point movement function

	Args:
	data (String): String data type under std_msgs.msg in ROS, data monitored by subscribing to the topic
	"""
	
	rospy.loginfo(f"callback_navigation_multipoint Received letter: {data.data}") 
	target_Multipoint_marker = data.data	#m1,m2,m3   /api/move?markers=m1,m2,m3&distance_tolerance=1.0&count=-1
	move_command = f'/api/move?markers={target_Multipoint_marker}&distance_tolerance=0.5&count=1'   #Call the mobile interface and move to the target point at location (15.0, 4.0, pai/2)
	chassis_client.send(move_command.encode('utf-8'))
	rospy.loginfo(f"send chassis_client :/api/move?markers={target_Multipoint_marker}&distance_tolerance=0.5&count=1")   

def callback_navigation_move_cancel(data: String):
	
	"""
	Callback function of topic/navigation_move_cancel

	Subscribe to/navigation_move_cancel to cancel the current movement instruction and stop the planner

	Args:
	data (String): String data type under std_msgs.msg in ROS, data monitored by subscribing to the topic
	"""
	
	rospy.loginfo(f"callback_navigation_move_cancel Received letter: {data.data}")
	move_command = f'/api/move/cancel'  
	chassis_client.send(move_command.encode('utf-8'))
	rospy.loginfo(f"send chassis_client :/api/move/cancel0&count=-1")   

def callback_get_robot_status(data: String):
	"""
	Callback function of topic/navigation_get_robot_status

	Obtain the current global status of the AGV chassis by subscribing to/navigation_get_robot_status

	Args:
	data (String): String data type under std_msgs.msg in ROS, data monitored by subscribing to the topic
	"""
	rospy.loginfo(f"callback_get_robot_status Received letter: {data.data}")
	move_command = f'/api/robot_status'	#Get the current global state of the robot
	chassis_client.send(move_command.encode('utf-8'))        
	rospy.loginfo(f"send chassis_client :/api/robot_status")   
	full_response = ''
	
	
def callback_navigation_position_adjust_marker(data: String):
	"""
	Callback function of topic /navigation_position_adjust_marker

	Correct the motion posture of the AGV chassis by subscribing to the topic /navigation_position_adjust_marker

	Args:
	data (String): String data type under std_msgs.msg in ROS, data monitored by subscribing to the topic
	"""
	
	rospy.loginfo(f"callback_navigation_position_adjust_marker Received letter: {data.data}")
	target_marker = data.data                                         #Specify a marker to correct the robot position
	move_command = f'/api/position_adjust?marker={target_marker}'   #Tell the robot that it is currently at the marker point codenamed 001
	chassis_client.send(move_command.encode('utf-8'))
	rospy.loginfo(f"send chassis_client :/api/position_adjust?marker={target_marker}")
	
def callback_get_power_status(data: String):

	"""
	Callback function of topic /navigation_get_power_status

	Obtain the power of the AGV chassis by subscribing to the topic /navigation_get_power_status

	Args:
	data (String): String data type under std_msgs.msg in ROS, data monitored by subscribing to the topic
	"""
	
	rospy.loginfo(f"callback_get_power_status Received letter: {data.data}")
	move_command = f'/api/get_power_status'                      #Get chassis power
	chassis_client.send(move_command.encode('utf-8'))
	rospy.loginfo(f"send chassis_client :/api/get_power_status")   
	full_response = ''
	
def callback_navigation_joy_control(data: NavigationJoyControl):
	"""
	Callback function of topic /navigation_joy_control

	Directly control the speed change on the x and y axes based on the global coordinate system of the machine by subscribing to the topic /navigation_joy_control

	Args:
	data (NavigationJoyControl): NavigationJoyControl data type under agv.msg in ROS, data monitored by subscribing to the topic
	"""
	
	rospy.loginfo(f"callback_navigation_joy_control Received letter: {data}")  #The robot can directly control the instructions, such as left turn, right turn, forward and backward
	angular_velocity = data.angular_velocity                    #The robot angular velocity setting range is (-1.0 ~ 1.0) rad/s. Positive: the robot turns left; negative: the robot turns right
	linear_velocity = data.linear_velocity                      #The robot linear speed setting range is (-0.5 ~ 0.5) m/s. Positive means the robot moves forward; negative means the robot moves backward.
	move_command = f'/api/joy_control?angular_velocity={angular_velocity}&linear_velocity={linear_velocity}'  
	chassis_client.send(move_command.encode('utf-8'))
	rospy.loginfo(f"send chassis_client :/api/joy_control?angular_velocity={angular_velocity}&linear_velocity={linear_velocity}")   

def callback_navigation_max_speed(data: Float64):
	"""
	Callback function of topic /navigation_max_speed

	Directly set the maximum speed of AGV by subscribing to the topic /navigation_max_speed, range [0.3, 0.7]

	Args:
	data (Float64): Float64 data type under std_msgs.msg in ROS, data monitored by subscribing to the topic
	"""
	
	rospy.loginfo(f"callback_navigation_max_speed Received letter: {data.data}")
	target = data.data
	move_command = f'/api/set_params?max_speed={target}'   #The robot's maximum travel speed (percentage) range is [0.3,0.7]
	chassis_client.send(move_command.encode('utf-8'))
	rospy.loginfo(f"send chassis_client :/api/set_params?max_speed={target}")

def callback_navigation_max_speed_ratio(data: Float64):
	"""
	Callback function of topic /navigation_max_speed_ratio

	Directly set the maximum speed ratio of AGV by subscribing to the topic /navigation_max_speed_ratio, range [0.3, 1.4]

	Args:
	data (Float64): Float64 data type under std_msgs.msg in ROS, data monitored by subscribing to the topic
	"""
	
	rospy.loginfo(f"callback_navigation_max_speed_ratio Received letter: {data.data}")
	target = data.data
	move_command = f'/api/set_params?max_speed_ratio={target}'     #The robot's maximum travel speed percentage range is [0.3,1.4]
	chassis_client.send(move_command.encode('utf-8'))
	rospy.loginfo(f"send chassis_client :/api/set_params?max_speed_ratio={target}")

def callback_navigation_max_speed_linear(data: Float64):
	"""
	Callback function of topic /navigation_max_speed_linear

	Directly set the maximum linear speed of AGV (m/s) by subscribing to the topic /navigation_max_speed_linear, range [0.1, 1.0]

	Args:
	data (Float64): Float64 data type under std_msgs.msg in ROS, data monitored by subscribing to the topic
	"""
	
	rospy.loginfo(f"callback_navigation_max_speed_linear Received letter: {data.data}")
	target = data.data
	move_command = f'/api/set_params?max_speed_linear={target}'    #Robot maximum linear speed (m/s) Value range [0.1,1.0]
	chassis_client.send(move_command.encode('utf-8'))               
	rospy.loginfo(f"send chassis_client :/api/set_params?max_speed_linear={target}")

def callback_navigation_max_speed_angular(data: Float64):
	"""
	Callback function of topic /navigation_max_speed_angular

	Directly set the maximum angular speed of AGV (rad/s) by subscribing to the topic /navigation_max_speed_angular, range [0.5,3.5]

	Args:
	data (Float64): Float64 data type under std_msgs.msg in ROS, data monitored by subscribing to the topic
	"""
	
	rospy.loginfo(f"callback_navigation_max_speed_angular Received letter: {data.data}")
	target = data.data
	move_command = f'/api/set_params?max_speed_angular={target}'     #Robot maximum angular velocity (rad/s) Value range [0.5,3.5]
	chassis_client.send(move_command.encode('utf-8'))
	rospy.loginfo(f"send chassis_client :/api/set_params?max_speed_angular={target}")

def callback_navigation_get_params(data: String):
	"""
	Callback function of topic/navigation_get_params

	Get the parameter list and current value by subscribing to the topic/navigation_get_params

	Args:
	data (String): String data type under std_msgs.msg in ROS, data listened by subscribing to the topic
	"""
	
	rospy.loginfo(f"callback_navigation_get_params Received letter: {data.data}") 
	move_command = f'/api/get_params'                               #Get parameter list and current value
	chassis_client.send(move_command.encode('utf-8')) 
	rospy.loginfo(f"send chassis_client :/api/get_params")	

def callback_navigation_LED_set_color(data: NavigationLedSetColor):
	"""
	Callback function of topic /navigation_LED_set_color

	Set the color of the light strip by subscribing to the topic /navigation_LED_set_color. The data range of the three RGB channels is [0, 100]

	Args:
	data (NavigationLedSetColor): NavigationLedSetColor data type under agv.msg in ROS,
	Data monitored by subscribing to the topic
	"""
	
	rospy.loginfo(f"callback_navigation_LED_set_color Received letter: {data}") 
	set_color_R = data.R
	set_color_G = data.G
	set_color_B = data.B
	move_command = f'/api/LED/set_color?r={set_color_R}&g={set_color_G}&b={set_color_B}'  #Set the light strip color to green /api/LED/set_color?r=100&g=100&b=0
	chassis_client.send(move_command.encode('utf-8')) 
	rospy.loginfo(f"send chassis_client :/api/LED/set_color?r={set_color_R}&g={set_color_G}&b={set_color_B}")		

def callback_get_current_map(data: String):
    """
    Callback function of topic /navigation_get_current_map

    Obtain the current map of the AGV chassis by subscribing to the topic /navigation_get_current_map

    Args:
    data (String): String data type under std_msgs.msg in ROS, data monitored by subscribing to the topic
    """
    rospy.loginfo(f"callback_get_current_map Received letter: {data.data}")
    move_command = f'/api/map/get_current_map'  # Get the current map
    chassis_client.send(move_command.encode('utf-8'))
    rospy.loginfo(f"send chassis_client :/api/map/get_current_map")


if __name__ == '__main__':
	
    chassis_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    chassis_host = '192.168.10.10'
    chassis_port = 31001
    chassis_client.connect((chassis_host, chassis_port))

    rospy.loginfo('---------------------------start----------------------------')

    rospy.init_node('process', anonymous=True)

    rospy.Subscriber("/navigation_marker", String, callback_navigation_marker)   # Single target point movement, move to the target point code "target_name"
    rospy.Subscriber("/navigation_location", NavigationLocation, callback_navigation_location) #Single target point movement, move to the target point at location (15.0, 4.0, pai/2)
    rospy.Subscriber("/navigation_multipoint", String, callback_navigation_multipoint)  #Multi-target point movement
    rospy.Subscriber("/navigation_move_cancel", String, callback_navigation_move_cancel) #Cancel the current move command
    rospy.Subscriber("/navigation_get_robot_status", String, callback_get_robot_status)   #Get the current global status of the robot
    rospy.Subscriber("/navigation_position_adjust_marker", String, callback_navigation_position_adjust_marker)  #Specify a marker to correct the robot position
    rospy.Subscriber("/navigation_get_power_status", String, callback_get_power_status)     #Get chassis power
    rospy.Subscriber("/navigation_joy_control", NavigationJoyControl, callback_navigation_joy_control)   #The robot can directly control the instructions, such as left turn, right turn, forward and backward
    rospy.Subscriber("/navigation_max_speed", Float64, callback_navigation_max_speed) #The robot's maximum travel speed (percentage) range is [0.3,0.7]
    rospy.Subscriber("/navigation_max_speed_ratio", Float64, callback_navigation_max_speed_ratio)  #The robot's maximum travel speed percentage range is [0.3,1.4]
    rospy.Subscriber("/navigation_max_speed_linear", Float64, callback_navigation_max_speed_linear) #Robot maximum linear speed (m/s) Value range [0.1,1.0]
    rospy.Subscriber("/navigation_max_speed_angular", Float64, callback_navigation_max_speed_angular) #Robot maximum angular velocity (rad/s) Value range [0.5,3.5]
    rospy.Subscriber("/navigation_get_params", String, callback_navigation_get_params) #Get parameter list and current value
    rospy.Subscriber("/navigation_LED_set_color", NavigationLedSetColor, callback_navigation_LED_set_color) #Set the light strip color
    rospy.Subscriber("/navigation_get_current_map", String, callback_get_current_map)  # Get the current map
    
    pub_navigation_feedback = rospy.Publisher('/navigation_feedback', String, queue_size=10)   

    while not rospy.is_shutdown():

        full_response = ''

        while not rospy.is_shutdown():

            try:
                part = chassis_client.recv(1024).decode()
                
                if len(part) > 2 :
					 
                    full_response += part
                    rospy.loginfo(f"part  recv  data: {part}")
                    pub_navigation_feedback.publish(part)   #The received information is published/navigation_feedback topic
                
                if full_response.count('\n') >= 3:  # Confirm that at least three JSON objects have been received
                    break
            
            except socket.timeout:

                rospy.logerr("Socket timeout while reading response")
                break

    rospy.loginfo('---------------------------final----------------------------')
    



    
