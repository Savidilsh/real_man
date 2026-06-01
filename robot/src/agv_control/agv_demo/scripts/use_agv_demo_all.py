#!/usr/bin/env python3
# -*- coding=UTF-8 -*-
"""
Copyright (c) 2024 [Ruiman Intelligent Technology Co., Ltd.]. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions of code in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This module AGV chassis ROS topic publishing example provides a good demonstration for secondary development.

This module provides [
single target preset point navigation, single target point navigation, multi-target preset point navigation, navigation cancellation, obtaining robot global status,
correcting robot posture, obtaining AGV chassis power, speed control, various maximum speed settings, obtaining parameter values, chassis RGB tricolor light settings
] function publisher demonstration. Through simple publishers, the corresponding functions of the chassis are tested one by one.

Example usage:
>>> pub_marker = rospy.Publisher('/navigation_marker', String, queue_size=10)
>>> pub_marker.publish(msg)
Other function functions are used in a similar way.
"""

import json

import rospy

from std_msgs.msg import String

from agv_ros.msg import NavigationLedSetColor
from agv_ros.msg import NavigationJoyControl
from agv_ros.msg import NavigationLocation

def callback_navigation_feedback(data):
    
    if data:
    	# Split and process each JSON object
        json_objects = data.data.split('\n')
        for json_str in json_objects:
            if json_str.strip():  # Ensure non-empty string
                json_data = json.loads(json_str)
                rospy.loginfo(f"Processed JSON: {json_data}")
                if  'command' in json_data: 
                    if  json_data['command'] == '/api/get_params':            # Example of judging whether get_params is successful
                        if 'results' in json_data :
                            rospy.loginfo(f"'max_speed_angular': {json_data['results']['max_speed_angular']}, 'max_speed_linear': {json_data['results']['max_speed_linear']}, 'max_speed_ratio': {json_data['results']['max_speed_ratio']}, 'status': {json_data['status']}")
                if 'code' in json_data :
                    if json_data['code'] == '01002':
                        rospy.loginfo("Move completed successfully.")

if __name__ == '__main__':
    
    rospy.init_node('navigation_publisher', anonymous=True)  # Initialize ROS node
    
    pub_LED_set_color = rospy.Publisher('/navigation_LED_set_color', NavigationLedSetColor, queue_size=10)
    pub_get_params = rospy.Publisher('/navigation_get_params', String, queue_size=10)
    pub_get_power_status = rospy.Publisher('/navigation_get_power_status', String, queue_size=10)  
    pub_joy_control = rospy.Publisher('/navigation_joy_control', NavigationJoyControl, queue_size=10)
    pub_marker = rospy.Publisher('/navigation_marker', String, queue_size=10)
    pub_move_cancel = rospy.Publisher('/navigation_move_cancel', String, queue_size=10)
    pub_location = rospy.Publisher('/navigation_location', NavigationLocation, queue_size=10)

    rospy.Subscriber("/navigation_feedback", String, callback_navigation_feedback)
    
    rospy.sleep(1)  # Waiting for publisher to initialize

    #Testing LEDs
    msg = NavigationLedSetColor()  # Creating a message object
    msg.R = 250  # Set the red component
    msg.G = 250  # Set the green component
    msg.B = 250  # Set the blue component

    pub_LED_set_color.publish(msg)  # Release setting led message
    rospy.loginfo("Published navigation LED color: R={}, G={}, B={}".format(msg.R, msg.G, msg.B))  
    
    #Test Get Parameters
    # pub_get_params.publish(" ")  # Publish a Get Parameters message
    # rospy.loginfo("Published navigation get params request")  
    
    #Test power acquisition
    # pub_get_power_status.publish(" ")  # Publish the message to obtain power
    # rospy.loginfo("Published navigation power status request")  
    
    
    #Testing chassis direct control
    # msg = NavigationJoyControl()  # Creating a message object
    # msg.angular_velocity = 0.0 #The robot angular velocity setting range is (-1.0 ~ 1.0) rad/s. Positive: the robot turns left; negative: the robot turns right
    # msg.linear_velocity = 0.1  #The robot linear speed setting range is (-0.5 ~ 0.5) m/s. Positive means the robot moves forward; negative means the robot moves backward.
    # pub_joy_control.publish(msg)  # Post a sports message
    # rospy.loginfo("Published navigation joy control command: angular={}, linear={}".format(msg.angular_velocity, msg.linear_velocity))  
    # rospy.sleep(2)
    
    #Test navigation to a marker
    # msg = String()  # Creating a message object
    # msg.data = 'point1'  # Set the message data to 'dianA'
    # pub_marker.publish(msg)  # Post a message to dianA
    # rospy.loginfo("Published navigation marker: {}".format(msg.data))  
    # rospy.sleep(15)
    
    # msg = NavigationLocation()
    # 15.0,4.0,1.5707963
    # msg.x = 15.0
    # msg.y = 4.0
    # msg.theta = 1.5707963
    # pub_location.publish(msg)
    # rospy.loginfo(f"Sending NavigationLocation: x={msg.x}, y={msg.y}, theta={msg.theta}") 
    # rospy.sleep(2)
    #Test navigation to a marker
    # pub_marker.publish("dianB")  # Post a message to dianB
    # rospy.loginfo("Published navigation marker: dianB")  
    # rospy.sleep(2)
    
    #Cancel motion during test navigation
    # pub_move_cancel.publish(" ")  # Post a Cancel Move message
    # rospy.loginfo("Published navigation move cancel command")  
    # rospy.sleep(2)
    
    
    
