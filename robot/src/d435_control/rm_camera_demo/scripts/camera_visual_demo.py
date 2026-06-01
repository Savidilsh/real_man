#!/usr/bin/env python3

# // Copyright RealMan
# // License(BSD/GPL/...)
# // Author: Haley
# // 整体demo完成对相机RGB图像和深度图像话题订阅，并通过OPENCV可视化图像信息。
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class NodeSubscribe:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('sub_image_node', anonymous=True)
        rospy.loginfo("这是RM的D435相机demo，利用opencv展示摄像头RGB和深度图像，按下ctrl+c退出当前demo")
        # 接收话题图像数据，并可视化
        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        self.color_image = None
        self.depth_image = None

    def color_callback(self, data):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

def main():
    node = NodeSubscribe()
    try:
        while not rospy.is_shutdown():
            if node.color_image is not None:
                cv2.imshow("color_frame", node.color_image)
            if node.depth_image is not None:
                cv2.imshow("depth_frame", node.depth_image)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
