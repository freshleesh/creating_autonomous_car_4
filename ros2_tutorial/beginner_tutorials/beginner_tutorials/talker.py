#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Int32


class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        # QoS options to try while testing behavior:
        # - history: HistoryPolicy.KEEP_LAST / HistoryPolicy.KEEP_ALL
        # - depth: queue size (used with KEEP_LAST)
        # - reliability: ReliabilityPolicy.RELIABLE / ReliabilityPolicy.BEST_EFFORT
        # - durability: DurabilityPolicy.VOLATILE / DurabilityPolicy.TRANSIENT_LOCAL
        #
        # Examples to experiment with:
        # 1) Drop-tolerant streaming:
        #    reliability=ReliabilityPolicy.BEST_EFFORT
        # 2) Late-joiner gets last sample:
        #    durability=DurabilityPolicy.TRANSIENT_LOCAL
        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher_ = self.create_publisher(Int32, 'counter', qos_profile)

        self.counter = 0
        self.timer_period = 1.0  # seconds (1 Hz)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info('Talker started: publishing Int32 on /counter at 1 Hz')

    def timer_callback(self):
        self.counter += 1
        msg = Int32()
        msg.data = self.counter
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    node = Talker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
