import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from f110_msgs.msg import WpntArray

from controller.estop import EStop

PARAMS = {
    'control_rate_hz':   50.0,
    'pp_lookahead':       1.0,    # kept for fallback; overridden by adaptive L_f
    'pp_wheelbase':       0.33,
    'pp_max_steer':       0.4,
    'pp_speed_boost':     1.5,
    'pp_clear_dist':      3.0,
    'pp_straight_kappa':  0.1,
    'pp_v_max_boost':     8.0,
    # Speed-adaptive lookahead: L_f = clip(gain * v, min, max)
    'pp_lookahead_gain':  0.35,
    'pp_lookahead_min':   0.8,
    'pp_lookahead_max':   2.5,
    # Curvature feedforward: delta += ff_gain * atan(L * kappa)
    'pp_ff_gain':         0.5,
    # CTE-adaptive lookahead: L_f shrinks when off-path → faster recovery
    # L_f = clip(gain * v / (1 + cte_gain*|CTE|), min, max)
    'pp_cte_gain':           2.0,
    # Direct error feedback (PD-like): on top of PP geometry
    'pp_Kp_cte':             0.3,  # lateral error gain [rad/m]
    'pp_K_heading':          0.4,  # heading error gain [rad/rad]
}


class PPNode(Node):

    def __init__(self):
        super().__init__('pp')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.estop          = EStop(self)
        self.lookahead      = p('pp_lookahead')
        self.wheelbase      = p('pp_wheelbase')
        self.max_steer      = p('pp_max_steer')
        self.speed_boost    = p('pp_speed_boost')
        self.clear_dist     = p('pp_clear_dist')
        self.straight_kappa = p('pp_straight_kappa')
        self.v_max_boost    = p('pp_v_max_boost')
        self.lookahead_gain = p('pp_lookahead_gain')
        self.lookahead_min  = p('pp_lookahead_min')
        self.lookahead_max  = p('pp_lookahead_max')
        self.ff_gain        = p('pp_ff_gain')
        self.cte_gain       = p('pp_cte_gain')
        self.Kp_cte         = p('pp_Kp_cte')
        self.K_heading      = p('pp_K_heading')

        self.scan      = None
        self.odom      = None
        self.waypoints = []

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        from sensor_msgs.msg import LaserScan
        self.create_subscription(LaserScan,  '/scan',             self._scan_cb, 10)
        self.create_subscription(Odometry,   '/vesc/odom',        self._odom_cb, 10)
        self.create_subscription(WpntArray,  '/global_waypoints', self._wp_cb, latched)
        self.drive_pub     = self.create_publisher(AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.lookahead_pub = self.create_publisher(Marker, '/pp/lookahead', 10)
        self.create_timer(1.0 / p('control_rate_hz'), self._loop)

        self.get_logger().info('PPNode ready')

    def _scan_cb(self, msg): self.scan = msg
    def _odom_cb(self, msg): self.odom = msg
    def _wp_cb(self, msg):   self.waypoints = msg.wpnts

    def _loop(self):
        if self.odom is None or not self.waypoints:
            return

        # estop.is_stop_required is not yet implemented; skip for PP
        steer, speed = self._compute()

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

    def _compute(self):
        p   = self.odom.pose.pose.position
        q   = self.odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        v   = abs(self.odom.twist.twist.linear.x)

        wp_xy = np.array([(w.x_m, w.y_m) for w in self.waypoints])

        # 1) transform waypoints to vehicle frame
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy  = wp_xy[:, 0] - p.x, wp_xy[:, 1] - p.y
        local_x = c * dx + s * dy   # forward component
        local_y = -s * dx + c * dy  # lateral component (left = positive)
        ahead   = local_x > 0

        # 2) nearest waypoint — used for CTE and feedforward kappa
        dists = np.hypot(dx, dy)
        nearest_idx = int(np.argmin(dists))

        # CTE: signed lateral distance to nearest waypoint in vehicle frame
        cte = local_y[nearest_idx]
        kappa_near = float(self.waypoints[nearest_idx].kappa_radpm)

        # Adaptive lookahead: shorter when off-path → faster recovery
        #   L_f = clip(gain*v / (1 + cte_gain*|CTE|), min, max)
        lookahead = float(np.clip(
            self.lookahead_gain * v / (
                1.0 + self.cte_gain * abs(cte)
            ),
            self.lookahead_min,
            self.lookahead_max,
        ))

        # lookahead target — used for PP steering geometry
        err = np.abs(dists - lookahead)
        err[~ahead] = np.inf
        if np.all(np.isinf(err)):
            target_idx = nearest_idx
        else:
            target_idx = int(np.argmin(err))

        self._publish_lookahead(self.waypoints[target_idx])

        # 3) bicycle pure pursuit
        gx = self.waypoints[target_idx].x_m
        gy = self.waypoints[target_idx].y_m
        lx = math.cos(-yaw) * (gx - p.x) - math.sin(-yaw) * (gy - p.y)
        ly = math.sin(-yaw) * (gx - p.x) + math.cos(-yaw) * (gy - p.y)
        L_f_sq = lx * lx + ly * ly
        if L_f_sq < 1e-6:
            return 0.0, max(float(self.waypoints[target_idx].vx_mps), 0.5)

        gamma = 2.0 * ly / L_f_sq
        delta = math.atan(self.wheelbase * gamma)

        # Curvature feedforward
        delta_ff = math.atan(self.wheelbase * kappa_near)
        delta += self.ff_gain * delta_ff

        # Heading + CTE direct feedback
        psi_ref = float(self.waypoints[nearest_idx].psi_rad)
        heading_err = math.atan2(math.sin(psi_ref - yaw), math.cos(psi_ref - yaw))
        delta += self.K_heading * heading_err + self.Kp_cte * cte

        delta = max(-self.max_steer, min(self.max_steer, delta))

        # 4) speed from waypoint velocity profile
        speed = float(self.waypoints[target_idx].vx_mps)
        if speed < 0.1:
            speed = 1.0

        # 5) boost speed on clear straights
        if abs(kappa_near) < self.straight_kappa and self._is_path_clear(self.clear_dist):
            speed = min(speed * self.speed_boost, self.v_max_boost)

        return delta, speed

    def _is_path_clear(self, dist_threshold):
        """Return True if forward scan sector (±30°) is obstacle-free beyond dist_threshold."""
        if self.scan is None:
            return False
        scan = self.scan
        n = len(scan.ranges)
        i_center = int(round((0.0 - scan.angle_min) / scan.angle_increment))
        i_half   = int(round(math.pi / 6.0 / scan.angle_increment))   # 30°
        i_lo = max(0, i_center - i_half)
        i_hi = min(n - 1, i_center + i_half)
        sector = np.array(scan.ranges[i_lo:i_hi + 1])
        sector = np.where(np.isfinite(sector), sector, 100.0)
        return float(sector.min()) > dist_threshold

    def _publish_lookahead(self, wp):
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns = 'pp_lookahead'; m.id = 0
        m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x = wp.x_m; m.pose.position.y = wp.y_m
        m.pose.position.z = 0.1;    m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.3
        m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0
        self.lookahead_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = PPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
