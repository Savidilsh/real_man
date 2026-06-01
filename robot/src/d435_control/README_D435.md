## **一.项目介绍**

本功能包是对d435相机的封装，通过话题可以得到颜色帧、深度帧和相应像素点的深度值。

## **二.代码结构**

```
d435_control
    ├── rm_camera_demo 获取相机颜色帧、深度帧和相应像素点的深度值的 demo 示例包
    │   ├── CMakeLists.txt
    │   ├── include
    │   ├── package.xml
    │   ├── scripts
    │   │   ├── camera_visual_demo.py 对相机RGB图像和深度图像话题订阅，并通过OPENCV可视化图像信息demo 示例
    │   │   └── show_center_coordinate.py 获取指定像素点的三维坐标 demo 示例
    │   └── src
    └── README_D435.md #功能包说明文件
```



## **三.RealSense的SDK2.0和ROS包安装**

1.注册服务器的公钥

- ```
  sudo apt-get update && sudo apt-get upgrade && sudo apt-get dist-upgrade
  sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
  ```

2.将服务器添加到存储库列表中

- ```
  sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
  ```

3.安装SDK2

- ```
  sudo apt-get install librealsense2-utils
  sudo apt-get install librealsense2-dev 
  ```

说明：如果在非Jetson Xavier NX设备上，建议用以下方式安装

方式一（apt）：
基础安装

- ```
  sudo apt-get install librealsense2-dkms
  sudo apt-get install librealsense2-utils
  ```

选装

- ```
  sudo apt-get install librealsense2-dev
  sudo apt-get install librealsense2-dbg
  ```

方式二（源代码）：
下载librealsense，然后进入该目录，运行下列指令安装和编译依赖项：

- ```
  sudo apt-get install libudev-dev pkg-config libgtk-3-dev
  sudo apt-get install libusb-1.0-0-dev pkg-config
  sudo apt-get install libglfw3-dev
  sudo apt-get install libssl-dev
  ```
  
- ```
  sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
  sudo udevadm control --reload-rules && udevadm trigger 
  mkdir build
  cd build
  cmake ../ -DBUILD_EXAMPLES=true
  make
  sudo make install
  ```

4.测试安装结果

- ```
  realsense-viewer 
  ```
  
  <img src="./../../images/image-1.png" alt="pic" style="zoom:50%;" />

5.安装ROS版本的realsense2_camera

- ```
  sudo apt-get install ros-$ROS_DISTRO-realsense2-camera
  sudo apt-get install ros-$ROS_DISTRO-realsense2-description
  ```

6.安装rgbd-launch

**rgbd_launch**是一组打开RGBD设备，并load 所有nodelets转化 raw depth/RGB/IR 流到深度图(depth image), 视差图(disparity image)和点云(point clouds)的launch文件集

- ```
  sudo apt-get install ros-noetic-rgbd-launch
  ```

测试：

- ```
  roslaunch realsense2_camera demo_pointcloud.launch 
  ```

​       <img src="./../../images/image-2.png" alt="pic" style="zoom:50%;" />

## **四.运行指令**

1.启动相机ros功能包的launch

打开新的**终端**需要进入到工作空间下source后启动程序

- ```
  cd ~/catkin_ws
  source devel/setup.bash
  roslaunch realsense2_camera rs_camera.launch
  ```

2.启动demo示例：

获取颜色帧demo 示例

- ```
  rosrun rm_camera_demo camera_visual_demo.py
  ```

获取深度帧demo 示例

- ```
  rosrun rm_camera_demo show_center_coordinate.py
  ```
