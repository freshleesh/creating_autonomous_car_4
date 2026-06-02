import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import WpntArray

PARAMS = {
    # timing
    'control_rate_hz':      50.0,

    # vehicle
    'stanley_wheelbase':     0.33,   # F1TENTH chassis [m]
    'stanley_max_steer':     0.4,    # mechanical steering limit [rad]
    'stanley_max_speed':     8.0,    # hard speed cap [m/s]
    'stanley_speed_boost':   1.2,    # multiplier on waypoint vx_mps

    # Stanley gains
    'stanley_k':             1.0,    # base cross-track gain
    'stanley_ks':            0.5,    # softening term — prevents div/0 at standstill
    'stanley_k_vel':         0.5,    # adaptive-k coefficient: k_eff = k / (1 + k_vel * v)
    'stanley_kd':            0.0,    # PD derivative gain (0 = disabled)
    'stanley_cte_alpha':     0.8,    # LPF smoothing for CTE derivative (higher = smoother)

    # speed limiting
    'stanley_a_lat_max':     3.0,    # max lateral acceleration [m/s²] for curvature-based speed cap

    # topics
    'odom_topic':           '/vesc/odom',   # override to /car_state/odom in sim mode
}


class StanleyNode(Node):
    """
    Stanley controller for F1TENTH.

    Steering law:
        δ = e_ψ + atan2(k_eff · e_d, ks + v) + kd · ė_d_lpf
          e_ψ       : heading error  (path yaw − car yaw), normalised to (−π, π]
          e_d       : signed cross-track error at front axle (+ = car right of path)
          k_eff     : velocity-adaptive gain = k / (1 + k_vel · v)
          ė_d_lpf   : LPF-filtered CTE derivative (noise suppression)

    Speed limiting:
        v_target = min(wp.vx_mps · boost,  sqrt(a_lat_max / |κ|),  max_speed)
        (a_lat = v² · κ  →  v_max = sqrt(a_lat_max / κ))
    """

    def __init__(self):
        super().__init__('stanley')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        self.k           = p('stanley_k')
        self.ks          = p('stanley_ks')
        self.k_vel       = p('stanley_k_vel')
        self.kd          = p('stanley_kd')
        self.cte_alpha   = p('stanley_cte_alpha')
        self.wheelbase   = p('stanley_wheelbase')
        self.max_steer   = p('stanley_max_steer')
        self.max_speed   = p('stanley_max_speed')
        self.speed_boost = p('stanley_speed_boost')
        self.a_lat_max   = p('stanley_a_lat_max')
        self.dt          = 1.0 / p('control_rate_hz')

        self.odom      = None
        self.waypoints = []

        # PD state
        self._prev_cte     = 0.0
        self._cte_dot_filt = 0.0

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(Odometry,  p('odom_topic'),      self._odom_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self._wp_cb, latched)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.create_timer(self.dt, self._loop)

        self.get_logger().info('StanleyNode ready')

    def _odom_cb(self, msg): self.odom = msg
    def _wp_cb(self, msg):   self.waypoints = msg.wpnts

    def _loop(self):
        if self.odom is None or not self.waypoints:
            return
        steer, speed = self._compute()
        msg = AckermannDriveStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = steer
        msg.drive.speed          = speed
        self.drive_pub.publish(msg)

    def _compute(self):
        pos = self.odom.pose.pose.position
        q   = self.odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # front axle position
        fx = pos.x + self.wheelbase * math.cos(yaw)
        fy = pos.y + self.wheelbase * math.sin(yaw)

        # nearest waypoint to front axle
        wp_x  = np.array([w.x_m  for w in self.waypoints])
        wp_y  = np.array([w.y_m  for w in self.waypoints])
        dists = np.hypot(wp_x - fx, wp_y - fy)
        idx   = int(np.argmin(dists))
        wp    = self.waypoints[idx]

        # heading error, normalised to (−π, π]
        e_psi = wp.psi_rad - yaw
        e_psi = math.atan2(math.sin(e_psi), math.cos(e_psi))

        # signed cross-track error at front axle
        dx  = fx - wp.x_m
        dy  = fy - wp.y_m
        e_d = math.sin(wp.psi_rad) * dx - math.cos(wp.psi_rad) * dy

        # actual vehicle speed for gain adaptation
        vx = abs(self.odom.twist.twist.linear.x)

        # velocity-adaptive k: softens gain at high speed to prevent oscillation
        k_eff = self.k / (1.0 + self.k_vel * vx)

        # PD derivative with LPF (noise suppression)
        cte_dot_raw        = (e_d - self._prev_cte) / self.dt
        self._cte_dot_filt = (self.cte_alpha * self._cte_dot_filt
                              + (1.0 - self.cte_alpha) * cte_dot_raw)
        self._prev_cte     = e_d
        pd_term            = self.kd * self._cte_dot_filt

        # curvature-based speed limit: a_lat = v² · |κ|  →  v ≤ sqrt(a_lat_max / |κ|)
        kappa    = abs(wp.kappa_radpm)
        v_kappa  = math.sqrt(self.a_lat_max / (kappa + 1e-6))

        # speed: waypoint profile * boost, capped by curvature and hard limit
        speed = float(wp.vx_mps) * self.speed_boost
        speed = max(speed, 0.5)
        speed = min(speed, v_kappa, self.max_speed)

        # Stanley steering law
        delta = e_psi + math.atan2(k_eff * e_d, self.ks + speed) + pd_term
        delta = float(np.clip(delta, -self.max_steer, self.max_steer))

        return delta, speed


def main(args=None):
    rclpy.init(args=args)
    node = StanleyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
