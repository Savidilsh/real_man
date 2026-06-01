<!-- filepath: /home/rm/catkin_ws/src/d435_control/README_EN.md -->
## **1. Project Introduction**

This package is an encapsulation of the d435 camera, which provides color frames, depth frames, and corresponding pixel depth values through topics.

## **2. Code Structure**

```
d435_control
    ├── rm_camera_demo Demo package for obtaining camera color frames, depth frames and corresponding pixel depth values
    │   ├── CMakeLists.txt
    │   ├── include
    │   ├── package.xml
    │   ├── scripts
    │   │   ├── camera_visual_demo.py Demo example for subscribing to camera RGB image and depth image topics and visualizing image information through OpenCV
    │   │   └── show_center_coordinate.py Demo example for obtaining 3D coordinates of specified pixel points
    │   └── src
    └── README_D435.md #Package description file
```

## **3. RealSense SDK 2.0 and ROS Package Installation**

1. Register the server's public key

- ```
  sudo apt-get update && sudo apt-get upgrade && sudo apt-get dist-upgrade
  sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
  ```

2. Add the server to the repository list

- ```
  sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
  ```

3. Install SDK 2

- ```
  sudo apt-get install librealsense2-utils
  sudo apt-get install librealsense2-dev 
  ```

Note: If you are using a device other than Jetson Xavier NX, it is recommended to install using the following methods:

Method 1 (apt):
Basic installation

- ```
  sudo apt-get install librealsense2-dkms
  sudo apt-get install librealsense2-utils
  ```

Optional installation

- ```
  sudo apt-get install librealsense2-dev
  sudo apt-get install librealsense2-dbg
  ```

Method 2 (from source):
Download librealsense, then navigate to that directory and run the following commands to install and compile dependencies:

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

4. Test the installation

- ```
  realsense-viewer 
  ```
  
  <img src="./../../images/image-1.png" alt="pic" style="zoom:50%;" />

5. Install the ROS version of realsense2_camera

- ```
  sudo apt-get install ros-$ROS_DISTRO-realsense2-camera
  sudo apt-get install ros-$ROS_DISTRO-realsense2-description
  ```

6. Install rgbd-launch

**rgbd_launch** is a set of launch files that open RGBD devices and load all nodelets to convert raw depth/RGB/IR streams to depth images, disparity images, and point clouds

- ```
  sudo apt-get install ros-noetic-rgbd-launch
  ```

Test:

- ```
  roslaunch realsense2_camera demo_pointcloud.launch 
  ```

   <img src="./../../images/image-2.png" alt="pic" style="zoom:50%;" />

## **4. Running Instructions**

1. Launch the camera ROS package

Open a new **terminal**, navigate to the workspace, source it, and then start the program:

- ```
  cd ~/catkin_ws
  source devel/setup.bash
  roslaunch realsense2_camera rs_camera.launch
  ```

2. Run demo examples:

Color frame demo example:

- ```
  rosrun rm_camera_demo camera_visual_demo.py
  ```

Depth frame demo example:

- ```
  rosrun rm_camera_demo show_center_coordinate.py
  ```