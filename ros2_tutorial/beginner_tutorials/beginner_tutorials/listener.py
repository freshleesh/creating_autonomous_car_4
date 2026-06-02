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


class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        # QoS options to try while testing behavior:
        # - history: HistoryPolicy.KEEP_LAST / HistoryPolicy.KEEP_ALL
        # - depth: queue size (used with KEEP_LAST)
        # - reliability: ReliabilityPolicy.RELIABLE / ReliabilityPolicy.BEST_EFFORT
        # - durability: DurabilityPolicy.VOLATILE / DurabilityPolicy.TRANSIENT_LOCAL
        #
        # NOTE:
        # - Publisher and subscriber QoS should be compatible to communicate.
        # - Intentionally mismatching settings (for example some durability cases)
        #   is useful to observe connection/data delivery differences.
        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(
            Int32,
            'counter',
            self.callback,
            qos_profile
        )
        self.get_logger().info('Listener started: subscribing to /counter')

    def callback(self, msg: Int32):
        self.get_logger().info(f'Received: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    node = Listener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
