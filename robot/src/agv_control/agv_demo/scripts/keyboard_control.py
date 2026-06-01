import sys
import rospy
from agv_ros.msg import NavigationJoyControl


def move_distance(x, y, pub):
    robot_step = 1
    robot_speed = 0.3
    msg = NavigationJoyControl()

    steps = x // robot_step
    for i in range(len(steps)):
        msg.linear_velocity(robot_speed)
        pub.publish(msg)
        rospy.sleep(0.5)


def main():
    rospy.init_node('keyboard_control')
    pub = rospy.Publisher('/navigation_joy_control', NavigationJoyControl, queue_size=10)
    rospy.sleep(1)  # 确保 Publisher 已初始化

    print("Use W/A/S/D to move, Q to quit.")

    while not rospy.is_shutdown():
        key = sys.stdin.read(1)  # 读取一个字符
        msg = NavigationJoyControl()
        
        if key.lower() == 'w':
            msg.linear_velocity = 0.3
        elif key.lower() == 's':
            msg.linear_velocity = -0.3
        elif key.lower() == 'a':
            msg.angular_velocity = 0.3
        elif key.lower() == 'd':
            msg.angular_velocity = -0.3
        elif key.lower() == 'q':
            print("Exiting...")
            break
        else:
            continue

        pub.publish(msg)
        print(f"Published: linear={msg.linear_velocity}, angular={msg.angular_velocity}")

if __name__ == '__main__':
    main()
