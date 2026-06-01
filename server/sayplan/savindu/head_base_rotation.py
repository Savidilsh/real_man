#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Head-to-base rotation helper.
"""

import json
import math
import threading
import time

import rospy
from std_msgs.msg import String
from agv_ros.msg import NavigationLocation
from servo_ros.msg import ServoMove

SERVO2_MIN = 350
SERVO2_MAX = 670
SERVO2_CENTER = 510
POSE_POLL_HZ = 2.0
UNITS_PER_DEG = 7
SETTLE_TIME = 0.3
STOP_DEG = 2.0
HEAD_HOLD_DELAY = 1.5 # Keep head at target angle before centering
BASE_START_DELAY = 0  # Delay after centering head before base command


class HeadBaseRotator:
    def __init__(self, external_theta_provider=None):
        """
        Initialize HeadBaseRotator.
        
        Args:
            external_theta_provider: Optional callable that returns (theta, timestamp) tuple.
                                   If provided, will use this instead of subscribing to feedback.
        """
        if not rospy.core.is_initialized():
            rospy.init_node("head_base_rotation", anonymous=True)
        self._pub_servo = rospy.Publisher("/servo_control/move", ServoMove, queue_size=10)
        self._pub_nav_location = rospy.Publisher("/navigation_location", NavigationLocation, queue_size=10)
        self._pub_get_status = rospy.Publisher("/navigation_get_robot_status", String, queue_size=1)
        
        self._external_theta_provider = external_theta_provider
        self._pose_lock = threading.Lock()
        self._current_theta = None
        self._current_x = None
        self._current_y = None
        self._pose_stamp = None
        self._face_servo2 = SERVO2_CENTER
        
        # Only subscribe if no external provider
        if external_theta_provider is None:
            self._sub_feedback = rospy.Subscriber("/navigation_feedback", String, self._feedback_cb, queue_size=10)
            self._poll_thread = threading.Thread(target=self._pose_poll_loop, daemon=True)
            self._poll_thread.start()
        else:
            self._sub_feedback = None
            self._poll_thread = None

    def _pose_poll_loop(self):
        rate = rospy.Rate(POSE_POLL_HZ)
        while not rospy.is_shutdown():
            self._pub_get_status.publish(String(data=""))
            rate.sleep()

    def _publish_servo2(self, angle):
        msg = ServoMove()
        msg.servo_id = 2
        msg.angle = int(angle)
        self._pub_servo.publish(msg)
        self._face_servo2 = int(angle)

    def _publish_nav_location(self, x, y, theta):
        msg = NavigationLocation()
        msg.x = float(x)
        msg.y = float(y)
        msg.theta = float(theta)
        self._pub_nav_location.publish(msg)

    def _feedback_cb(self, msg):
        if not msg or not msg.data:
            return
        try:
            json_objects = msg.data.strip().split("\n")
            for json_str in json_objects:
                if not json_str:
                    continue
                data = json.loads(json_str)
                command = data.get("command") or data.get("commanWd")
                if command == "/api/robot_status" and "results" in data:
                    pose_data = data["results"].get("current_pose")
                    if not pose_data:
                        continue
                    theta = pose_data.get("theta")
                    x = pose_data.get("x")
                    y = pose_data.get("y")
                    if theta is None:
                        continue
                    with self._pose_lock:
                        self._current_theta = float(theta)
                        self._current_x = float(x) if x is not None else None
                        self._current_y = float(y) if y is not None else None
                        self._pose_stamp = time.time()
        except Exception:
            return

    def _get_theta_snapshot(self):
        # Use external provider if available
        if self._external_theta_provider is not None:
            return self._external_theta_provider()
        
        # Otherwise use internal subscription
        with self._pose_lock:
            if self._current_theta is not None:
                return float(self._current_theta), self._pose_stamp
        return None, None

    def _wait_for_theta(self, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline and not rospy.is_shutdown():
            theta, _ = self._get_theta_snapshot()
            if theta is not None:
                return theta
            time.sleep(0.05)
        return None

    def _normalize_angle(self, rad):
        return (rad + math.pi) % (2 * math.pi) - math.pi

    def rotate_to_servo2(self, target_servo2):
        """Rotate base until head-servo offset is achieved using navigation_location.

        Publishes one NavigationLocation with current (x, y) and target theta,
        then waits for theta to reach the goal (listen-only).
        """
        target_servo2 = int(target_servo2)
        if target_servo2 < SERVO2_MIN or target_servo2 > SERVO2_MAX:
            raise ValueError(f"servo2 out of range {SERVO2_MIN}-{SERVO2_MAX}: {target_servo2}")

        self._publish_servo2(target_servo2)
        if SETTLE_TIME > 0:
            time.sleep(SETTLE_TIME)

        # Keep head at requested angle before centering
        if HEAD_HOLD_DELAY > 0:
            time.sleep(HEAD_HOLD_DELAY)

        if target_servo2 == SERVO2_CENTER:
            self._face_servo2 = SERVO2_CENTER
            return

        delta_deg = (target_servo2 - SERVO2_CENTER) / UNITS_PER_DEG
        start_yaw = self._wait_for_theta()
        if start_yaw is None:
            raise RuntimeError("Failed to read base yaw from /navigation_feedback")

        with self._pose_lock:
            x = self._current_x
            y = self._current_y
        if x is None or y is None:
            raise RuntimeError("Failed to read base position from /navigation_feedback")

        target_yaw = start_yaw + math.radians(delta_deg)

        # Center head before base motion
        self._publish_servo2(SERVO2_CENTER)
        if BASE_START_DELAY > 0:
            time.sleep(BASE_START_DELAY)

        self._publish_nav_location(x, y, target_yaw)

        # Listen for completion (theta close to target)
        deadline = time.time() + 30.0
        rate = rospy.Rate(POSE_POLL_HZ)
        while not rospy.is_shutdown():
            theta, _ = self._get_theta_snapshot()
            if theta is None:
                raise RuntimeError("Lost /navigation_feedback pose during rotation")

            if abs(math.degrees(self._normalize_angle(theta - target_yaw))) <= STOP_DEG:
                break

            if time.time() > deadline:
                raise RuntimeError("NavigationLocation rotation timeout")
            rate.sleep()

        self._face_servo2 = SERVO2_CENTER

    def rotate_base_by_deg(self, delta_deg, use_feedback=True):
        """Rotate base by degrees. Positive = right, negative = left.

        Publishes one NavigationLocation with current (x, y) and target theta,
        then waits for theta to reach the goal (listen-only).

        Args:
            delta_deg: Degrees to rotate
            use_feedback: Ignored (kept for compatibility)
        """
        delta_deg = float(delta_deg)
        if abs(delta_deg) <= 0.0:
            return

        self._publish_servo2(SERVO2_CENTER)
        if SETTLE_TIME > 0:
            time.sleep(SETTLE_TIME)

        if HEAD_HOLD_DELAY > 0:
            time.sleep(HEAD_HOLD_DELAY)

        start_yaw = self._wait_for_theta()
        if start_yaw is None:
            raise RuntimeError("Failed to read initial base yaw from /navigation_feedback")

        with self._pose_lock:
            x = self._current_x
            y = self._current_y
        if x is None or y is None:
            raise RuntimeError("Failed to read base position from /navigation_feedback")

        target_yaw = start_yaw + math.radians(delta_deg)

        # Ensure head is centered during base motion
        self._publish_servo2(SERVO2_CENTER)
        if BASE_START_DELAY > 0:
            time.sleep(BASE_START_DELAY)

        self._publish_nav_location(x, y, target_yaw)

        deadline = time.time() + 30.0
        rate = rospy.Rate(POSE_POLL_HZ)
        while not rospy.is_shutdown():
            theta, _ = self._get_theta_snapshot()
            if theta is None:
                raise RuntimeError("Lost /navigation_feedback pose during rotation")

            if abs(math.degrees(self._normalize_angle(theta - target_yaw))) <= STOP_DEG:
                break

            if time.time() > deadline:
                raise RuntimeError("NavigationLocation rotation timeout")
            rate.sleep()

        self._face_servo2 = SERVO2_CENTER


def main():
    rotator = HeadBaseRotator()
    rospy.sleep(1.0)

    print("Head-base rotation ready. Enter servo2 angle or 'quit'.")
    while not rospy.is_shutdown():
        try:
            text = input("servo2> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in {"q", "quit", "exit"}:
            break
        try:
            target = int(text)
        except ValueError:
            print("Invalid number.")
            continue
        try:
            rotator.rotate_to_servo2(target)
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
