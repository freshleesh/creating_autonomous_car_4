import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Header

from controller.estop import EStop

PARAMS = {
    'control_rate_hz':  50.0,
    'gf_bubble_radius':  0.3,
    'gf_speed':          2.0,
    'gf_max_steer':      0.4,
    'gf_max_range':     10.0,
}


class GapFollowNode(Node):

    def __init__(self):
        super().__init__('gap_follow')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.estop         = EStop(self)
        self.bubble_radius = p('gf_bubble_radius')
        self.speed         = p('gf_speed')
        self.max_steer     = p('gf_max_steer')
        self.max_range     = p('gf_max_range')
        self.fov           = math.radians(180)  # front field-of-view to consider for gap detection [rad]
        self.speed = 3.0      #2.0 / 1.2      1.5
        self.free_threshold = 1.5  # minimum distance to consider a gap as free [m]
        self.dt = 0.002
        self.pre_error = 0.0
        self.p_gain = 1.0
        self.d_gain = 0.1

        self.scan = None
        self.odom = None

        self.create_subscription(LaserScan, '/scan',      self._scan_cb, 10)
        self.create_subscription(Odometry,  '/vesc/odom', self._odom_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.gap_pub = self.create_publisher(MarkerArray, '/gaps', 10)

        self.create_timer(1.0 / p('control_rate_hz'), self._loop)

        self.get_logger().info('GapFollowNode ready')

    def _scan_cb(self, msg): self.scan = msg
    def _odom_cb(self, msg): self.odom = msg

    def _loop(self):
        if self.scan is None or self.odom is None:
            return

        steer, speed = self._compute()

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

    def _compute(self):
        # TODO: Implement Follow-the-Gap (FTG) using LiDAR free-space selection
        #
        # Goal:
        #   - Use LiDAR scan data to find a collision-free gap
        #   - Select a safe target direction inside the best gap
        #   - Convert the target direction into a steering command
        #
        # You should return (steering, speed) from this function.
        #
        # Useful information:
        #   - self.scan.ranges             : LiDAR distance array [m]
        #   - self.scan.angle_min          : angle of first beam [rad]
        #   - self.scan.angle_increment    : angular step between beams [rad]
        #   - self.bubble_radius           : safety bubble radius around closest obstacle [m]
        #   - self.max_range               : cap LiDAR ranges to suppress outliers [m]
        #   - self.max_steer               : steering clamp [rad]
        #   - self.speed                   : nominal driving speed [m/s]
        #   - self.odom                    : current vehicle motion (optional for speed adjustment)
        #
        # Suggested approach (FTG pipeline):
        #   - Preprocess ranges:
        #       * replace NaN/Inf/invalid values
        #       * clip ranges to [0, self.max_range]
        #       * optionally focus on a front field-of-view
        #   - Find the closest obstacle beam
        #   - Create a safety bubble around that obstacle (zero-out nearby beams)
        #   - Find the longest contiguous non-zero gap
        #   - Choose the best target beam in the gap
        #       * e.g., farthest beam or weighted by distance and heading
        #   - Convert target beam index to steering angle
        #   - Clamp steering to +/- self.max_steer
        #   - Optionally reduce speed when |steering| is large or obstacle is close
        #
        # Output:
        #   - steering [rad]
        #   - speed [m/s]

        free_threshold = self.free_threshold  # [m]
        ranges = np.array(self.scan.ranges)
        # NaN/Inf → 10.0
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        angles = self.scan.angle_min + np.arange(len(ranges)) * self.scan.angle_increment

        # Front FOV only 
        front_mask = (angles >= -self.fov / 2) & (angles <= self.fov / 2)
        front_ranges = ranges[front_mask]
        front_angles = angles[front_mask]

        # Free if range > threshold
        is_free = front_ranges > free_threshold

        # Inflate obstacles: zero out neighbors near any obstacle beam
        bubble_indices = 0
        for i in range(len(is_free)):
            if front_ranges[i] < free_threshold:
                lo = max(0, i - bubble_indices)
                hi = min(len(is_free), i + bubble_indices + 1)
                is_free[lo:hi] = False

        # Find contiguous free gaps
        gaps = []
        start = None
        for i in range(len(is_free)):
            if is_free[i] and start is None:
                start = i
            elif not is_free[i] and start is not None:
                gaps.append((start, i - 1))
                start = None
        if start is not None:
            gaps.append((start, len(is_free) - 1))

        # Find the longest gap
        longest_gap = None
        longest_len = 0
        for s, e in gaps:
            if e - s + 1 > longest_len:
                longest_len = e - s + 1
                longest_gap = (s, e)

        # --- Visualization ---
        stamp = self.get_clock().now().to_msg()
        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)
        marker_id = 0

        # 1) Radius free_threshold circle (white)
        circle_marker = Marker()
        circle_marker.header.frame_id = 'laser'
        circle_marker.header.stamp = stamp
        circle_marker.ns = 'threshold_circle'
        circle_marker.id = marker_id
        marker_id += 1
        circle_marker.type = Marker.LINE_STRIP
        circle_marker.action = Marker.ADD
        circle_marker.scale.x = 0.03
        circle_marker.color.a = 0.6
        circle_marker.color.r = 1.0
        circle_marker.color.g = 1.0
        circle_marker.color.b = 1.0
        for a in np.linspace(-self.fov / 2, self.fov / 2, 100):
            circle_marker.points.append(Point(
                x=float(free_threshold * math.cos(a)),
                y=float(free_threshold * math.sin(a)),
                z=0.0))
        marker_array.markers.append(circle_marker)

        # 2) All gaps as filled sectors (TRIANGLE_LIST), r=1.0
        sector_r = 10.0
        origin = Point(x=0.0, y=0.0, z=0.0)
        for s, e in gaps:
            is_longest = (longest_gap is not None and s == longest_gap[0] and e == longest_gap[1])
            fan_marker = Marker()
            fan_marker.header.frame_id = 'laser'
            fan_marker.header.stamp = stamp
            fan_marker.ns = 'gap_fans'
            fan_marker.id = marker_id
            marker_id += 1
            fan_marker.type = Marker.TRIANGLE_LIST
            fan_marker.action = Marker.ADD
            fan_marker.scale.x = 1.0
            fan_marker.scale.y = 1.0
            fan_marker.scale.z = 1.0
            fan_marker.color.a = 0.5
            if is_longest:
                fan_marker.color.r = 1.0
            else:
                fan_marker.color.g = 1.0
            for idx in range(s, e):
                a0 = front_angles[idx]
                a1 = front_angles[idx + 1]
                p0 = Point(x=float(sector_r * math.cos(a0)),
                           y=float(sector_r * math.sin(a0)), z=0.0)
                p1 = Point(x=float(sector_r * math.cos(a1)),
                           y=float(sector_r * math.sin(a1)), z=0.0)
                fan_marker.points.append(origin)
                fan_marker.points.append(p0)
                fan_marker.points.append(p1)
            marker_array.markers.append(fan_marker)

        # 3) Target heading line + P control
        target_angle = 0.0
        if longest_gap is not None:
            s, e = longest_gap
            mid_idx = (s + e) // 2
            target_angle = float(front_angles[mid_idx])

        # Red line toward target
        line_marker = Marker()
        line_marker.header.frame_id = 'laser'
        line_marker.header.stamp = stamp
        line_marker.ns = 'target_line'
        line_marker.id = marker_id
        marker_id += 1
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.05
        line_marker.color.a = 1.0
        line_marker.color.r = 1.0
        line_marker.points.append(Point(x=0.0, y=0.0, z=0.0))
        line_marker.points.append(Point(
            x=float(10.0 * math.cos(target_angle)),
            y=float(10.0 * math.sin(target_angle)),
            z=0.0))
        marker_array.markers.append(line_marker)

        self.gap_pub.publish(marker_array)

        # P control: steering = target_angle, clamped
        error = target_angle
        error_div = (error - self.pre_error) / self.dt
        steer = error * self.p_gain + error * self.d_gain
        self.pre_error = error
        steer = np.clip(target_angle, -self.max_steer, self.max_steer)
        
        self.get_logger().info('Target speed: {:.2f} m/s, Steering command: {:.2f} deg'.format(
            self.speed, math.degrees(steer)))

        return float(steer), self.speed


def main(args=None):
    rclpy.init(args=args)
    node = GapFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
