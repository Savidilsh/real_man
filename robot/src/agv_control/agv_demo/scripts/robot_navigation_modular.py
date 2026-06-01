#!/usr/bin/env python3
# -*- coding=UTF-8 -*-
"""
模块化机器人导航示例

- RobotStateMonitor: 负责获取机器人当前位置和朝向
- RobotMover: 负责根据delta distance和delta angle计算目标位姿并导航

使用方法:
1. 确保agv_controller.py正在运行
2. 运行此脚本: python3 robot_navigation_modular.py
"""

import json
import math
import rospy
from std_msgs.msg import String
from agv_ros.msg import NavigationLocation

class RobotStateMonitor:
    """
    负责获取机器人当前位置和朝向
    """
    def __init__(self):
        rospy.init_node('robot_state_monitor', anonymous=True)
        self.pub_robot_status = rospy.Publisher('/navigation_get_robot_status', String, queue_size=10)
        self.sub_feedback = rospy.Subscriber("/navigation_feedback", String, self._callback_navigation_feedback)
        self._robot_status = None
        rospy.sleep(1)

    def _callback_navigation_feedback(self, data):
        if data and data.data:
            try:
                json_objects = data.data.split('\n')
                for json_str in json_objects:
                    if json_str.strip():
                        json_data = json.loads(json_str)
                        if 'command' in json_data and json_data['command'] == '/api/robot_status':
                            self._robot_status = json_data
            except Exception as e:
                rospy.logerr(f"处理反馈信息时出错: {e}")

    def request_robot_status(self):
        msg = String()
        msg.data = ""
        self.pub_robot_status.publish(msg)
        rospy.loginfo("请求获取机器人状态...")

    def get_current_pose(self):
        if self._robot_status and 'results' in self._robot_status:
            results = self._robot_status['results']
            if 'current_pose' in results:
                pose = results['current_pose']
                return {
                    'x': pose.get('x'),
                    'y': pose.get('y'),
                    'theta': pose.get('theta')
                }
        return None

class RobotMover:
    """
    负责根据delta distance和delta angle计算目标位姿并导航
    """
    def __init__(self, state_monitor: RobotStateMonitor):
        self.state_monitor = state_monitor
        self.pub_location = rospy.Publisher('/navigation_location', NavigationLocation, queue_size=10)
        self.navigation_completed = False
        self.sub_feedback = rospy.Subscriber("/navigation_feedback", String, self._callback_navigation_feedback)
        rospy.sleep(1)

    def _callback_navigation_feedback(self, data):
        if data and data.data:
            try:
                json_objects = data.data.split('\n')
                for json_str in json_objects:
                    if json_str.strip():
                        json_data = json.loads(json_str)
                        if 'code' in json_data and json_data['code'] == '01002':
                            rospy.loginfo("导航任务完成!")
                            self.navigation_completed = True
            except Exception as e:
                rospy.logerr(f"处理反馈信息时出错: {e}")

    def move_relative(self, delta_distance, delta_angle):
        """
        相对运动：先转向再前进
        1. 如果delta_angle不为0，先转向相应角度
        2. 如果delta_distance不为0，再前进相应距离
        3. 每次导航都有超时保护，超时则取消
        """
        # 添加导航取消发布器
        self.pub_cancel = rospy.Publisher('/navigation_move_cancel', String, queue_size=10)
        
        # 第一步：处理转向
        if abs(delta_angle) > 0.001:  # 角度不为0
            rospy.loginfo(f"开始转向 {delta_angle:.2f}°")
            
            # 计算转向目标位置（原地转向）
            pose = self.state_monitor.get_current_pose()
            if pose is None:
                rospy.logwarn("无法获取当前位置，不能导航")
                return False
                
            x, y, theta = pose['x'], pose['y'], pose['theta']
            delta_angle_rad = math.radians(delta_angle)
            target_theta = (theta + delta_angle_rad) % (2 * math.pi)
            
            rospy.loginfo(f"转向前朝向: {math.degrees(theta):.2f}°")
            rospy.loginfo(f"转向目标朝向: {math.degrees(target_theta):.2f}°")
            
            # 发送转向命令
            msg = NavigationLocation()
            msg.x = x  # 保持当前位置
            msg.y = y  # 保持当前位置
            msg.theta = target_theta  # 只改变朝向
            self.pub_location.publish(msg)
            self.navigation_completed = False
            
            rospy.loginfo(f"转向目标: theta={math.degrees(target_theta):.2f}°")
            
            # 等待转向完成
            if not self.wait_for_navigation_completion(timeout=10.0):
                rospy.logwarn("转向超时，取消导航")
                self.pub_cancel.publish(String(""))
                return False
            else:
                rospy.loginfo("转向完成")
            
            # 转向完成后重新获取机器人状态，确保获取到正确的朝向
            rospy.sleep(0.5)  # 等待状态更新
            self.state_monitor.request_robot_status()
            rospy.sleep(0.5)  # 等待状态返回
        
        # 第二步：处理前进
        if abs(delta_distance) > 0.001:  # 距离不为0
            rospy.loginfo(f"开始前进 {delta_distance:.2f}m")
            
            # 获取当前位置（转向后的位置）
            pose = self.state_monitor.get_current_pose()
            if pose is None:
                rospy.logwarn("无法获取当前位置，不能导航")
                return False
                
            x, y, theta = pose['x'], pose['y'], pose['theta']
            
            rospy.loginfo(f"前进前朝向: {math.degrees(theta):.2f}°")
            
            # 计算前进目标位置
            target_x = x + delta_distance * math.cos(theta)
            target_y = y + delta_distance * math.sin(theta)
            
            # 发送前进命令 - 明确指定目标朝向为当前朝向
            msg = NavigationLocation()
            msg.x = target_x
            msg.y = target_y
            msg.theta = theta  # 明确指定目标朝向为当前朝向，防止导航系统重新计算
            self.pub_location.publish(msg)
            self.navigation_completed = False
            
            rospy.loginfo(f"前进目标: x={target_x:.4f}, y={target_y:.4f}, theta={math.degrees(theta):.2f}°")
            
            # 等待前进完成
            if not self.wait_for_navigation_completion(timeout=10.0):
                rospy.logwarn("前进超时，取消导航")
                self.pub_cancel.publish(String(""))
                return False
            else:
                rospy.loginfo("前进完成")
        
        # 显示最终位置
        final_pose = self.state_monitor.get_current_pose()
        if final_pose:
            rospy.loginfo(f"最终位置: x={final_pose['x']:.4f}, y={final_pose['y']:.4f}, theta={math.degrees(final_pose['theta']):.2f}°")
        
        rospy.loginfo(f"相对运动完成: 距离={delta_distance:.2f}m, 角度={delta_angle:.2f}°")
        return True

    def wait_for_navigation_completion(self, timeout=30.0):
        start_time = rospy.get_time()
        while not self.navigation_completed and not rospy.is_shutdown():
            if rospy.get_time() - start_time > timeout:
                rospy.logwarn("导航超时!")
                return False
            rospy.sleep(0.1)
        return self.navigation_completed

if __name__ == '__main__':
    try:
        monitor = RobotStateMonitor()
        mover = RobotMover(monitor)
        rospy.loginfo("模块化导航示例已启动")
        rospy.loginfo("按 Ctrl+C 退出")
        
        # 获取初始位置
        monitor.request_robot_status()
        rospy.sleep(2)

        # 测试新的分步导航逻辑
        # 示例1：先转向90度，再前进1米
        rospy.loginfo("=== 测试1：转向90度，再前进1米 ===")
        if mover.move_relative(5.0, -180.0):
            rospy.loginfo("测试1完成!")
        else:
            rospy.logwarn("测试1失败!")
        
        rospy.sleep(3)
        exit()
        
        # # 示例2：只转向-45度
        # rospy.loginfo("=== 测试2：只转向-45度 ===")
        # if mover.move_relative(0.0, -45.0):
        #     rospy.loginfo("测试2完成!")
        # else:
        #     rospy.logwarn("测试2失败!")
        
        # rospy.sleep(3)
        
        # # 示例3：只前进0.5米
        # rospy.loginfo("=== 测试3：只前进0.5米 ===")
        # if mover.move_relative(0.5, 0.0):
        #     rospy.loginfo("测试3完成!")
        # else:
        #     rospy.logwarn("测试3失败!")

        # 实时显示当前位置
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            pose = monitor.get_current_pose()
            if pose:
                rospy.loginfo(f"当前位置: x={pose['x']:.4f}, y={pose['y']:.4f}, theta={math.degrees(pose['theta']):.2f}°")
            rate.sleep()

    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户中断")
    except Exception as e:
        rospy.logerr(f"程序运行出错: {e}") 