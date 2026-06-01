#!/usr/bin/env python3

# // Copyright RealMan
# // License(BSD/GPL/...)
# // Author: Haley
# // 整体demo完成对图像（640*480）中心像素点获取三维坐标。首先通过ROS订阅图像话题和相机参数话题，获取图像数据和相机内参，通过相机内参对齐深度图像和RGB图像，得到可靠的中心像素点深度，再根据内参和深度算出三维坐标，同时可视化深度图像并打印坐标。

import cv2
import pyrealsense2 as rs2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
import numpy as np


class pic_pix:
    '''
    像素坐标选定
    '''
    def __init__(self):
        self.pix_x = 320
        self.pix_y = 240

class NodeSubscribe:
    '''
    订阅相机图像话题和相机参数话题,通过相机内参对齐深度图像和RGB图像，得到可靠的中心像素点深度，再根据内参和深度算出三维坐标，同时可视化深度图像并打印坐标。
    '''
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('Center_Coordinate_node', anonymous=True)
        rospy.loginfo("这是RM的D435相机demo，利用深度对齐图像获取图像中心点坐标值（640*480），按下ctrl+c退出当前demo")
        self.sub = rospy.Subscriber('/camera/depth/image_rect_raw', msg_Image, self.imageDepthCallback)  # 接收图像话题数据
        self.sub_info = rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.imageDepthInfoCallback)  # 接收相机参数话题数据
        self.intrinsics = None
        self.pix = pic_pix()

    def imageDepthCallback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)  # 图像格式转换，ROS话题格式转为CV格式
            if self.intrinsics:
                depth = cv_image[self.pix.pix_y, self.pix.pix_x]
                result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [self.pix.pix_x, self.pix.pix_y], depth)  # 计算像素点三维坐标
                print('图像中心点坐标值(mm):', result)
            cv2.imshow("aligned_depth_to_color_frame", cv_image)
            key = cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return

    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.intrinsics:
                return
            print(type(cameraInfo))
            print(cameraInfo)
            print(self.intrinsics)
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.K[2]
            self.intrinsics.ppy = cameraInfo.K[5]
            self.intrinsics.fx = cameraInfo.K[0]
            self.intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.D]
        except CvBridgeError as e:
            print(e)
            return


def main():
    node = NodeSubscribe()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

