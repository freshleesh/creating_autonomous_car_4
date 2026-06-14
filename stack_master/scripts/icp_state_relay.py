#!/usr/bin/env python3
"""ICP pose + VESC twist -> /car_state/odom (EKF 대체 relay).

EKF 융합의 평활/지연 없이 ICP 위치를 그대로 쓰되, 컨트롤러/플래너가 필요로 하는
속도(twist: linear.x, angular.z)는 /vesc/odom 에서 가져와 합친다. 즉 위치는 순수
ICP, 속도는 휠 오도메트리.

선택적으로 map->base_link TF 도 발행해 "RViz로 보는 TF == 차가 쓰는 /car_state/odom"
을 보장한다(실차). 시뮬에서는 gym_bridge 가 TF 를 소유하므로 publish_tf=false 로 둔다.

  subscribes : /icp/pose/odom (nav_msgs/Odometry, map 프레임 pose)
               /vesc/odom     (nav_msgs/Odometry, base_link 프레임 twist)
  publishes  : /car_state/odom (nav_msgs/Odometry, pose=ICP + twist=VESC)
               map->base_link TF (publish_tf=true 일 때만)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class IcpStateRelay(Node):
    def __init__(self):
        super().__init__('icp_state_relay')

        self.icp_topic = self.declare_parameter('icp_topic', '/icp/pose/odom').value
        self.vesc_topic = self.declare_parameter('vesc_topic', '/vesc/odom').value
        self.out_topic = self.declare_parameter('output_topic', '/car_state/odom').value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.publish_tf = self.declare_parameter('publish_tf', True).value

        self.last_twist = None  # geometry_msgs/TwistWithCovariance, 최신 VESC

        self.pub = self.create_publisher(Odometry, self.out_topic, 10)
        # VESC odom 은 보통 best-effort(SensorDataQoS) 로 나온다 -> 매칭.
        self.create_subscription(Odometry, self.vesc_topic, self._vesc_cb, qos_profile_sensor_data)
        # ICP pose 는 reliable 로 발행됨.
        self.create_subscription(Odometry, self.icp_topic, self._icp_cb, 10)

        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.get_logger().info(
            f'icp_state_relay up: pose<-{self.icp_topic}  twist<-{self.vesc_topic}  '
            f'-> {self.out_topic}  publish_tf={self.publish_tf}')

    def _vesc_cb(self, msg: Odometry):
        self.last_twist = msg.twist

    def _icp_cb(self, msg: Odometry):
        out = Odometry()
        out.header = msg.header          # map 프레임 + ICP 타임스탬프
        out.header.frame_id = self.map_frame
        out.child_frame_id = self.base_frame
        out.pose = msg.pose              # 순수 ICP pose (+ 공분산)
        if self.last_twist is not None:
            out.twist = self.last_twist  # VESC 속도 (linear.x, angular.z)
        self.pub.publish(out)

        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = msg.header.stamp
            tf.header.frame_id = self.map_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = msg.pose.pose.position.x
            tf.transform.translation.y = msg.pose.pose.position.y
            tf.transform.translation.z = 0.0
            tf.transform.rotation = msg.pose.pose.orientation
            self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = IcpStateRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
