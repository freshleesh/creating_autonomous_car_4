#!/usr/bin/env python3
"""Print the yaw (deg) of an Odometry/PoseStamped topic — for checking IMU sign.

Usage:
  ros2 run stack_master yaw_watch.py --ros-args -p topic:=/icp/pose/odom
  # or just:  python3 yaw_watch.py /icp/pose/odom

Rotate the car LEFT (counter-clockwise) and watch yaw INCREASE. If it decreases,
set imu_yaw_scale: -1.0 in icp.yaml / ndt.yaml.
"""
import math
import sys

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def yaw_deg(q):
    return math.degrees(2.0 * math.atan2(q.z, q.w))


class YawWatch(Node):
    def __init__(self, topic):
        super().__init__('yaw_watch')
        self.get_logger().info(f'Watching yaw on {topic}  (rotate LEFT -> yaw should increase)')
        self.create_subscription(Odometry, topic, self.cb, 10)

    def cb(self, msg):
        gz = msg.twist.twist.angular.z
        print(f'yaw = {yaw_deg(msg.pose.pose.orientation):+7.2f} deg', flush=True)


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else '/icp/pose/odom'
    rclpy.init()
    rclpy.spin(YawWatch(topic))


if __name__ == '__main__':
    main()
