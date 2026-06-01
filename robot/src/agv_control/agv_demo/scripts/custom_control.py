import sys
import rospy
from agv_ros.msg import NavigationJoyControl
import math

def move_distance(x, y, pub):

    distance = math.sqrt(x**2 + y**2)
    step_dis = 0.5
    msg = NavigationJoyControl()

    theta = math.atan2(y,x) * 180/math.pi
    # robot_speed = 0.3 if theta>=0 and theta < 180 else -0.3
    robot_speed = 0.3
    print(theta, robot_speed)
    steps = int(distance / step_dis)
    for i in range(steps):
        msg.linear_velocity = robot_speed
        pub.publish(msg)
        rospy.sleep(0.3)

    # angular_step = 1 if y>=0 else -1
    # msg.angular_velocity = angular_step
    # pub.publish(msg)
    # rospy.sleep(0.3)
    

def main():
    rospy.init_node('custom_control')
    pub = rospy.Publisher('/navigation_joy_control', NavigationJoyControl, queue_size=10)
    rospy.sleep(1)  # 确保 Publisher 已初始化
    move_distance(-3, 0, pub)

    # while not rospy.is_shutdown():
        # key = sys.stdin.read(1)  # 读取一个字符
        # msg = NavigationJoyControl()
        
        # move_distance(3, 0, pub)
        # if key.lower() == 'w':
        #     msg.linear_velocity = 0.3
        # elif key.lower() == 's':
        #     msg.linear_velocity = -0.3
        # elif key.lower() == 'a':
        #     msg.angular_velocity = 0.3
        # elif key.lower() == 'd':
        #     msg.angular_velocity = -0.3
        # elif key.lower() == 'q':
        #     print("Exiting...")
        #     break
        # else:
        #     continue

        # pub.publish(msg)
        # print(f"Published: linear={msg.linear_velocity}, angular={msg.angular_velocity}")

if __name__ == '__main__':
    main()
