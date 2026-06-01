## **一.项目介绍**

本功能包是对云迹底盘的ros封装，通过发布对应话题可以执行对应命令。

## **二.代码结构**

agv_control
    ├── agv_demo
    │   ├── CMakeLists.txt
    │   ├── include
    │   │   └── agv_demo
    │   ├── package.xml
    │   ├── scripts
    │   │   ├── use_agv_demo_all.py  #功能包功能的使用案例1
    │   │   └── use_agv_demo.py      #功能包功能的使用案例2，二次开发的示例
    │   └── src
    ├── agv_ros
    │   ├── CMakeLists.txt
    │   ├── include
    │   │   └── agv
    │   ├── launch
    │   │   └── agv_start.launch #底盘ros功能包的launch启动文件
    │   ├── msg                  #自定义的消息类型
    │   │   ├── navigation_joy_control.msg
    │   │   ├── navigation_LED_set_color.msg
    │   │   └── navigation_location.msg
    │   ├── package.xml
    │   ├── scripts
    │   │   └── agv_controller.py #封装的底盘ros功能包
    │   └── src
    └── README.md                 #功能包说明文件



## **三.编译方法**

- 创建 文件夹
    - mkdir -p ~/catkin_ws/src
- 将 agv_control文件夹放入工作空间catkin_ws/src/中
- 编译ros包

    - cd ~/catkin_ws
    - catkin build 

## **四.运行指令**

- 1.启动底盘ros功能包的launch
'''
cd ~/catkin_ws
source devel/setup.bash
roslaunch agv_ros agv_start.launch
'''

- 2.启动功能包功能的使用案例1或者2，比如案例1：
'''
rosrun agv_demo use_agv_demo.py
'''

## **五.注意**
    运行的示例代码的时候需要注意，底盘移动到标记点之前需要先建图并设置标记点。
    （建图和设置标记点教程见链接：http://waterdocs.pages.yunjichina.com.cn/user_manual/exports/WATER%EF%BC%88%E6%B0%B4%E6%BB%B4%EF%BC%89%E7%94%A8%E6%88%B7%E4%BD%BF%E7%94%A8%E6%89%8B%E5%86%8C-v0.5.1.pdf）

