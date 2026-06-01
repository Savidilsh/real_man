#!/usr/bin/env python3
# -*- coding=UTF-8 -*-
"""
版权所有 (c) 2024 [睿尔曼智能科技有限公司]。保留所有权利。
作者: Robert 时间: 2024/07/20

在满足以下条件的情况下，允许重新分发和使用源代码和二进制形式的代码，无论是否修改：
1. 重新分发的源代码必须保留上述版权声明、此条件列表和以下免责声明。
2. 以二进制形式重新分发的代码必须在随分发提供的文档和/或其他材料中复制上述版权声明、此条件列表和以下免责声明。

本软件由版权持有者和贡献者“按原样”提供，不提供任何明示或暗示的保证，
包括但不限于对适销性和特定用途适用性的暗示保证。
在任何情况下，即使被告知可能发生此类损害的情况下，
版权持有者或贡献者也不对任何直接的、间接的、偶然的、特殊的、惩罚性的或后果性的损害
（包括但不限于替代商品或服务的采购；使用、数据或利润的损失；或业务中断）负责，
无论是基于合同责任、严格责任还是侵权行为（包括疏忽或其他原因）。

此模块通过函数的形式实现每个功能的发布者模块的封装，只需要调用函数便可对相应功能进行测试。

此模块提供[
    单目标预设点导航、单目标点位导航、多目标预设点导航、导航取消、获取机器人全局状态、
    矫正机器人位姿、获取AGV底盘电量、速度控制、各项最大速度设置、获取参数值、底盘RGB三色灯设置
]的话题发布功能。等待用户调用，进而测试相应模块功能。

示例用法：
>>> publish_navigation_get_robot_status()
其他函数功能使用方式类似。
"""

import rospy

from std_msgs.msg import String
from std_msgs.msg import Float64

from agv_ros.msg import NavigationLedSetColor
from agv_ros.msg import NavigationLocation  
from agv_ros.msg import NavigationJoyControl


def publish_navigation_led_color():
    """单次测试RGB三色灯的发布"""   
    
    rospy.init_node('navigation_led_publisher', anonymous=True)  # 初始化ROS节点
    pub = rospy.Publisher('/navigation_LED_set_color', NavigationLedSetColor, queue_size=10)  # 定义发布器，指定话题和消息类型
    rospy.sleep(1)  # 等待发布器初始化

    msg = NavigationLedSetColor()  # 创建消息对象
    msg.R = 250  # 设置红色分量
    msg.G = 250  # 设置绿色分量
    msg.B = 250  # 设置蓝色分量

    pub.publish(msg)  # 发布消息
    rospy.loginfo("Published navigation LED color: R={}, G={}, B={}".format(msg.R, msg.G, msg.B))  # 打印日志信息

def publish_navigation_get_params():
    """单次测试导航相关参数的发布"""

    
    rospy.init_node('navigation_get_params_publisher', anonymous=True)  # 初始化ROS节点
    pub = rospy.Publisher('/navigation_get_params', String, queue_size=10)  # 定义发布器，指定话题和消息类型
    rospy.sleep(1)  # 等待发布器初始化

    msg = String()  # 创建消息对象
    msg.data = ''  # 设置消息数据为空字符串

    pub.publish(msg)  # 发布消息
    rospy.loginfo("Published navigation get params request")  # 打印日志信息


def publish_navigation_power_status():
    """单次测试AGV底盘电量数据的发布""" 
    
    rospy.init_node('navigation_power_status_publisher', anonymous=True)  # 初始化ROS节点
    pub = rospy.Publisher('/navigation_get_power_status', String, queue_size=10)  # 定义发布器，指定话题和消息类型
    rospy.sleep(1)  # 等待发布器初始化

    msg = String()  # 创建消息对象
    msg.data = ''  # 设置消息数据为空字符串

    pub.publish(msg)  # 发布消息
    rospy.loginfo("Published navigation power status request")  # 打印日志信息

def publish_navigation_multipoint():
    """单次测试多点位导航的发布"""
    
    rospy.init_node('navigation_multipoint_publisher', anonymous=True)  # 初始化ROS节点
    pub = rospy.Publisher('/navigation_multipoint', String, queue_size=10)  # 定义发布器，指定话题和消息类型
    rospy.sleep(1)  # 等待发布器初始化

    msg = String()  # 创建消息对象
    msg.data = 'dianA,dianB,dianC'  # 设置消息数据为指定的字符串

    pub.publish(msg)  # 发布消息
    rospy.loginfo("Published navigation multipoint command: {}".format(msg.data))  # 打印日志信息

def publish_navigation_get_robot_status():
    """单次测试AGV状态的发布"""

    rospy.init_node('navigation_get_robot_status_publisher', anonymous=True)  # 初始化ROS节点
    pub = rospy.Publisher('/navigation_get_robot_status', String, queue_size=10)  # 定义发布器，指定话题和消息类型
    rospy.sleep(1)  # 等待发布器初始化

    msg = String()  # 创建消息对象
    msg.data = ''  # 设置消息数据为空字符串

    pub.publish(msg)  # 发布消息
    rospy.loginfo("Published navigation get robot status request")  # 打印日志信息

def publish_navigation_joy_control():
    """单次测试AGV底盘控制速度数据的发布"""
    rospy.init_node('navigation_joy_control_publisher', anonymous=True)  # 初始化ROS节点
    pub = rospy.Publisher('/navigation_joy_control', NavigationJoyControl, queue_size=10)  # 定义发布器，指定话题和消息类型
    rospy.sleep(1)  # 等待发布器初始化

    msg = NavigationJoyControl()  # 创建消息对象
    msg.angular_velocity = 0.0  #机器人角速度设值范围为 (-1.0 ~ 1.0)rad/s  正 机器人原地左转;负 机器人原地右转
    msg.linear_velocity = 0.0  #机器人线速度设值范围为 (-0.5 ~ 0.5)m/s  正  机器人前进 ;负 机器人后退

    pub.publish(msg)  # 发布消息
    rospy.loginfo("Published navigation joy control command: angular={}, linear={}".format(msg.angular_velocity, msg.linear_velocity))  # 打印日志信息

def publish_navigation_location():
    
    rospy.init_node('navigation_location_publisher', anonymous=True)  # 初始化ROS节点
    pub = rospy.Publisher('/navigation_location', NavigationLocation, queue_size=10)  # 定义发布器，指定话题和消息类型
    rospy.sleep(1)  # 等待发布器初始化

    msg = NavigationLocation()  # 创建消息对象
    msg.x = 0.0  # 设置x坐标
    msg.y = 0.0  # 设置y坐标
    msg.theta = 0.0  # 设置角度

    pub.publish(msg)  # 发布消息
    rospy.loginfo("Published navigation location: x={}, y={}, theta={}".format(msg.x, msg.y, msg.theta))  # 打印日志信息

def publish_navigation_marker():
    """单次测试单点导航的发布"""

    rospy.init_node('navigation_marker_publisher', anonymous=True)  # 初始化ROS节点
    pub = rospy.Publisher('/navigation_marker', String, queue_size=10)  # 定义发布器，指定话题和消息类型
    rospy.sleep(1)  # 等待发布器初始化

    msg = String()  # 创建消息对象
    msg.data = 'dianA'  # 设置消息数据为 'dianA'

    pub.publish(msg)  # 发布消息
    rospy.loginfo("Published navigation marker: {}".format(msg.data))  # 打印日志信息
    
def publish_navigation_max_speed():
    """单次测试AGV底盘最大速度数据的发布"""

    rospy.init_node('navigation_max_speed_publisher', anonymous=True)  # 初始化ROS节点
    pub = rospy.Publisher('/navigation_max_speed', Float64, queue_size=10)  # 定义发布器，指定话题和消息类型
    rospy.sleep(1)  # 等待发布器初始化

    msg = Float64()  # 创建消息对象
    msg.data = 0.5  #机器人最大行进速度(百分比) 取值范围[0.3,0.7]

    pub.publish(msg)  # 发布消息
    rospy.loginfo("Published navigation max speed: {}".format(msg.data))  # 打印日志信息

def publish_navigation_max_speed_angular():
    """单次测试AGV底盘最大航偏角速度数据的发布"""

    rospy.init_node('navigation_max_speed_angular_publisher', anonymous=True)  # 初始化ROS节点
    pub = rospy.Publisher('/navigation_max_speed_angular', Float64, queue_size=10)  # 定义发布器，指定话题和消息类型
    rospy.sleep(1)  # 等待发布器初始化

    msg = Float64()  # 创建消息对象
    msg.data = 0.8  #机器人最大行进速度百分比 取值范围[0.3,1.4] 

    pub.publish(msg)  # 发布消息
    rospy.loginfo("Published navigation max angular speed: {}".format(msg.data))  # 打印日志信息

def publish_navigation_max_speed_linear():
    """单次测试AGV底盘最大线速度数据的发布"""

    rospy.init_node('navigation_max_speed_linear_publisher', anonymous=True)  # 初始化ROS节点
    pub = rospy.Publisher('/navigation_max_speed_linear', Float64, queue_size=10)  # 定义发布器，指定话题和消息类型
    rospy.sleep(1)  # 等待发布器初始化

    msg = Float64()  # 创建消息对象
    msg.data = 0.8   #机器人最大线速度(m/s) 取值范围[0.1,1.0] 

    pub.publish(msg)  # 发布消息
    rospy.loginfo("Published navigation max linear speed: {}".format(msg.data))  # 打印日志信息

def publish_navigation_max_speed_ratio():
    """单次测试AGV底盘速度数据比的发布"""
    
    rospy.init_node('navigation_max_speed_ratio_publisher', anonymous=True)  # 初始化ROS节点
    pub = rospy.Publisher('/navigation_max_speed_ratio', Float64, queue_size=10)  # 定义发布器，指定话题和消息类型
    rospy.sleep(1)  # 等待发布器初始化

    msg = Float64()  # 创建消息对象
    msg.data = 1.2  #机器人最大角速度(rad/s) 取值范围[0.5,3.5] 

    pub.publish(msg)  # 发布消息
    rospy.loginfo("Published navigation max speed ratio: {}".format(msg.data))  # 打印日志信息

def publish_navigation_move_cancel():
    """单次测试导航停止规划命令的发布"""
    
    rospy.init_node('navigation_move_cancel_publisher', anonymous=True)  # 初始化ROS节点
    pub = rospy.Publisher('/navigation_move_cancel', String, queue_size=10)  # 定义发布器，指定话题和消息类型
    rospy.sleep(1)  # 等待发布器初始化

    msg = String()  # 创建消息对象
    msg.data = ''  # 设置消息数据为空字符串，取消移动命令

    pub.publish(msg)  # 发布消息
    rospy.loginfo("Published navigation move cancel command")  # 打印日志信息

def publish_navigation_position_adjust_marker():
    """单次测试导航点位调整消息的发布"""

    rospy.init_node('navigation_position_adjust_marker_publisher', anonymous=True)  # 初始化ROS节点
    pub = rospy.Publisher('/navigation_position_adjust_marker', String, queue_size=10)  # 定义发布器，指定话题和消息类型
    rospy.sleep(1)  # 等待发布器初始化

    msg = String()  # 创建消息对象
    msg.data = 'dianA'  # 设置消息数据为 'dianA'，调整位置标记

    pub.publish(msg)  # 发布消息
    rospy.loginfo("Published navigation position adjust marker: {}".format(msg.data))  # 打印日志信息

if __name__ == '__main__':

    try:
        #e.g. 测试获取状态
        publish_navigation_get_robot_status()
    except rospy.ROSInterruptException:
        pass




    
