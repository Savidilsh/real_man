#!/usr/bin/env python3
# -*- coding=UTF-8 -*-
"""
获取机器人状态的示例脚本

这个脚本展示了如何通过ROS话题获取AGV机器人的当前状态，
包括位置、朝向等信息。

使用方法:
1. 确保agv_controller.py正在运行
2. 运行此脚本: python3 get_robot_state_example.py
"""

import json
import rospy
from std_msgs.msg import String

class RobotStateMonitor:
    def __init__(self):
        """初始化机器人状态监控器"""
        rospy.init_node('robot_state_monitor', anonymous=True)
        
        # 发布器 - 用于请求机器人状态
        self.pub_robot_status = rospy.Publisher('/navigation_get_robot_status', String, queue_size=10)
        
        # 存储最新的机器人状态
        self.robot_status = None
        
        rospy.sleep(1)  # 等待发布器和订阅器初始化
        
    def callback_navigation_feedback(self, data):
        """
        处理来自机器人的反馈信息
        
        Args:
            data (String): 机器人返回的JSON格式状态信息
        """
        if data and data.data:
            try:
                # 分割并处理每个JSON对象
                json_objects = data.data.split('\n')
                for json_str in json_objects:
                    if json_str.strip():  # 确保非空字符串
                        json_data = json.loads(json_str)
                        
                        # 处理机器人状态信息
                        if 'command' in json_data and json_data['command'] == '/api/robot_status':
                            self.robot_status = json_data
                            
                            # 解析位置信息
                            if 'results' in json_data and 'current_pose' in json_data['results']:
                                pose = json_data['results']['current_pose']
                                x = pose.get('x', 'N/A')
                                y = pose.get('y', 'N/A')
                                theta = pose.get('theta', 'N/A')
                                
                                rospy.loginfo(f"机器人位置: x={x}, y={y}, theta={theta}")
                                
            except json.JSONDecodeError as e:
                rospy.logwarn(f"JSON解析错误: {e}")
            except Exception as e:
                rospy.logerr(f"处理反馈信息时出错: {e}")
    
    def get_robot_status(self):
        """请求获取机器人状态"""
        rospy.loginfo("请求获取机器人状态...")
        msg = String()
        msg.data = ""
        self.pub_robot_status.publish(msg)
    
    def get_current_position(self):
        """获取当前机器人位置和朝向"""
        if self.robot_status and 'results' in self.robot_status:
            results = self.robot_status['results']
            if 'current_pose' in results:
                pose = results['current_pose']
                return {
                    'x': pose.get('x'),
                    'y': pose.get('y'),
                    'theta': pose.get('theta')
                }
        return None

def main():
    """主函数"""
    try:
        # 创建机器人状态监控器
        monitor = RobotStateMonitor()
        
        rospy.loginfo("机器人状态监控器已启动")
        rospy.loginfo("按 Ctrl+C 退出")
        
        # 定期获取机器人状态
        rate = rospy.Rate(1)  # 1Hz
        
        while not rospy.is_shutdown():
            # 每5秒获取一次机器人状态
            if int(rospy.get_time()) % 5 == 0:
                monitor.get_robot_status()
            
            # 获取并显示当前位置
            position = monitor.get_current_position()
            if position:
                rospy.loginfo(f"当前位置: x={position['x']}, y={position['y']}, theta={position['theta']}")
            
            rate.sleep()
            
    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户中断")
    except Exception as e:
        rospy.logerr(f"程序运行出错: {e}")

if __name__ == '__main__':
    main() 