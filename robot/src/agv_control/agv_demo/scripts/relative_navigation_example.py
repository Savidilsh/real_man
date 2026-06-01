#!/usr/bin/env python3
# -*- coding=UTF-8 -*-
"""
相对导航示例脚本

这个脚本展示了如何使用相对导航功能：
1. 实时获取机器人位置
2. 执行相对导航（距离+角度）
3. 等待导航完成

使用方法:
1. 确保agv_controller.py正在运行
2. 运行此脚本: python3 relative_navigation_example.py
"""

import rospy
import math
from get_robot_state_example import RobotStateMonitor

def main():
    """主函数"""
    try:
        # 创建机器人状态监控器
        monitor = RobotStateMonitor()
        
        rospy.loginfo("相对导航示例已启动")
        rospy.loginfo("按 Ctrl+C 退出")
        
        # 获取初始位置
        monitor.get_robot_status()
        rospy.sleep(2)
        
        # 示例1: 前进1米，不改变朝向
        rospy.loginfo("=== 示例1: 前进1米 ===")
        if monitor.navigate_to_target_location(1.0, 0.0):
            if monitor.wait_for_navigation_completion():
                rospy.loginfo("前进完成!")
            else:
                rospy.logwarn("前进未完成")
        rospy.sleep(2)
        
        # 示例2: 原地左转30度
        rospy.loginfo("=== 示例2: 原地左转30度 ===")
        if monitor.navigate_to_target_location(0.0, 30.0):
            if monitor.wait_for_navigation_completion():
                rospy.loginfo("左转完成!")
            else:
                rospy.logwarn("左转未完成")
        rospy.sleep(2)
        
        # 示例3: 前进0.5米，同时右转15度
        rospy.loginfo("=== 示例3: 前进0.5米，右转15度 ===")
        if monitor.navigate_to_target_location(0.5, -15.0):
            if monitor.wait_for_navigation_completion():
                rospy.loginfo("复合运动完成!")
            else:
                rospy.logwarn("复合运动未完成")
        rospy.sleep(2)
        
        # 示例4: 后退0.3米，左转45度
        rospy.loginfo("=== 示例4: 后退0.3米，左转45度 ===")
        if monitor.navigate_to_target_location(-0.3, 45.0):
            if monitor.wait_for_navigation_completion():
                rospy.loginfo("后退+左转完成!")
            else:
                rospy.logwarn("后退+左转未完成")
        rospy.sleep(2)
        
        rospy.loginfo("所有示例执行完成!")
        
        # 保持运行，显示实时位置
        rate = rospy.Rate(1)  # 1Hz
        while not rospy.is_shutdown():
            position = monitor.get_current_position()
            if position:
                rospy.loginfo(f"当前位置: x={position['x']:.4f}, y={position['y']:.4f}, theta={math.degrees(position['theta']):.2f}°")
            rate.sleep()
            
    except rospy.ROSInterruptException:
        rospy.loginfo("程序被用户中断")
    except Exception as e:
        rospy.logerr(f"程序运行出错: {e}")

if __name__ == '__main__':
    main() 