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
    'pp_lookahead':       1.0,
    'pp_wheelbase':       0.33,
    'pp_max_steer':       0.4,
    'pp_speed_boost':     1.5,
    'pp_clear_dist':      3.0,
    'pp_straight_kappa':  0.1,
    'pp_v_max_boost':     8.0,
    # Speed-adaptive lookahead
    'pp_lookahead_gain':  0.35,
    'pp_lookahead_min':   0.8,
    'pp_lookahead_max':   2.5,
    # Curvature feedforward
    'pp_ff_gain':         0.5,
    # CTE-adaptive lookahead
    'pp_cte_gain':        3.0,   # raised: shrink L_f more aggressively when off-path
    # Direct error feedback
    'pp_Kp_cte':          0.3,
    'pp_K_heading':       0.4,
    # [v2] Steering rate limiter: max steering change per timestep (rad/s)
    'pp_steer_rate_max':  2.0,
    # [v2] Preview speed horizon
    'pp_preview_n':       20,    # raised: look further ahead for corners
    # [v2] Preview blend ratio: 0=only target speed, 1=only preview min speed
    'pp_preview_blend':   0.65,  # raised: trust preview more
    # [v2] Speed smoothing low-pass alpha (0=no smoothing, 1=frozen)
    'pp_speed_alpha':     0.2,
    # [v2] Recovery mode: triggered when |CTE| > threshold
    'pp_cte_recovery_threshold': 0.6,   # (m) enter recovery above this
    'pp_recovery_speed':         0.8,   # (m/s) crawl speed during recovery
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
        self.steer_rate_max       = p('pp_steer_rate_max')
        self.preview_n            = int(p('pp_preview_n'))
        self.preview_blend        = p('pp_preview_blend')
        self.speed_alpha          = p('pp_speed_alpha')
        self.cte_recovery_thresh  = p('pp_cte_recovery_threshold')
        self.recovery_speed       = p('pp_recovery_speed')

        self.scan      = None
        self.odom      = None
        self.waypoints = []

        # [v2] state carried across timesteps
        self._prev_target_idx = 0       # avoids jumping backward on the track
        self._prev_steer      = 0.0     # for steering rate limiter
        self._smoothed_speed  = None    # for speed low-pass filter
        self._dt = 1.0 / p('control_rate_hz')

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        from sensor_msgs.msg import LaserScan
        self.create_subscription(LaserScan,  '/scan',             self._scan_cb, 10)
        self.create_subscription(Odometry,   '/vesc/odom',        self._odom_cb, 10)
        self.create_subscription(WpntArray,  '/global_waypoints', self._wp_cb, latched)
        from visualization_msgs.msg import MarkerArray
        self.drive_pub        = self.create_publisher(AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.lookahead_pub    = self.create_publisher(Marker,      '/pp/lookahead',        10)
        self.cte_pub          = self.create_publisher(Marker,      '/pp/cte_line',         10)
        self.circle_pub       = self.create_publisher(Marker,      '/pp/lookahead_circle', 10)
        self.status_pub       = self.create_publisher(Marker,      '/pp/status_text',      10)
        self.create_timer(self._dt, self._loop)

        self.get_logger().info('PPNode v2 ready')

    def _scan_cb(self, msg): self.scan = msg
    def _odom_cb(self, msg): self.odom = msg
    def _wp_cb(self, msg):
        self.waypoints = msg.wpnts
        self._prev_target_idx = 0   # reset on new waypoint set

    def _loop(self):
        if self.odom is None or not self.waypoints:
            return

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
        N = len(wp_xy)

        # 1) vehicle frame transform
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy  = wp_xy[:, 0] - p.x, wp_xy[:, 1] - p.y
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        ahead   = local_x > 0

        dists = np.hypot(dx, dy)

        # 2) nearest waypoint — global search (no window restriction)
        #    always find the true nearest point to support recovery
        nearest_idx = int(np.argmin(dists))

        cte = local_y[nearest_idx]
        kappa_near = float(self.waypoints[nearest_idx].kappa_radpm)

        # ── RECOVERY MODE ────────────────────────────────────────────────────
        # Only trigger when no valid ahead waypoints exist globally.
        # CTE-based recovery removed — caused spinning at corners.
        in_recovery = False
        valid_ahead = np.where(ahead)[0]
        if len(valid_ahead) == 0:
            in_recovery = True

        if in_recovery:
            # Find nearest ahead-ish waypoint by relaxing the ahead constraint:
            # pick the globally nearest waypoint and use PP geometry toward it.
            self._prev_target_idx = nearest_idx
            self._publish_lookahead(self.waypoints[nearest_idx])
            self._publish_viz(p, yaw, self.lookahead_min,
                              self.waypoints[nearest_idx], cte, True)

            psi_ref = float(self.waypoints[nearest_idx].psi_rad)
            heading_err = math.atan2(math.sin(psi_ref - yaw),
                                     math.cos(psi_ref - yaw))
            # Gentle heading correction only — no aggressive 1.5x that caused spinning
            delta = 0.4 * heading_err
            delta = max(-self.max_steer, min(self.max_steer, delta))

            max_delta_change = self.steer_rate_max * self._dt
            delta = float(np.clip(delta,
                                  self._prev_steer - max_delta_change,
                                  self._prev_steer + max_delta_change))
            self._prev_steer = delta
            self._smoothed_speed = self.recovery_speed
            self.get_logger().warn(
                f'[PP] RECOVERY(no-ahead): heading_err={math.degrees(heading_err):.1f}°',
                throttle_duration_sec=0.5)
            return delta, self.recovery_speed
        # ── END RECOVERY ─────────────────────────────────────────────────────

        # 3) Adaptive lookahead (CTE-shrinking) — normal mode
        lookahead = float(np.clip(
            self.lookahead_gain * v / (1.0 + self.cte_gain * abs(cte)),
            self.lookahead_min,
            self.lookahead_max,
        ))

        # target waypoint — search forward from nearest, avoid backward jumps
        forward_window = min(120, N - 1)
        fwd_indices = np.arange(nearest_idx, nearest_idx + forward_window) % N
        fwd_ahead   = ahead[fwd_indices]
        fwd_err     = np.abs(dists[fwd_indices] - lookahead)
        fwd_err[~fwd_ahead] = np.inf

        # fwd_err all-inf can't happen here (recovery mode caught it above)
        target_idx = int(fwd_indices[np.argmin(fwd_err)])
        self._prev_target_idx = target_idx
        self._publish_lookahead(self.waypoints[target_idx])
        self._publish_viz(p, yaw, lookahead,
                          self.waypoints[nearest_idx], cte, False)

        # 4) bicycle pure pursuit
        gx = self.waypoints[target_idx].x_m
        gy = self.waypoints[target_idx].y_m
        lx = math.cos(-yaw) * (gx - p.x) - math.sin(-yaw) * (gy - p.y)
        ly = math.sin(-yaw) * (gx - p.x) + math.cos(-yaw) * (gy - p.y)
        L_f_sq = lx * lx + ly * ly
        if L_f_sq < 1e-6:
            return self._prev_steer, max(float(self.waypoints[target_idx].vx_mps), 0.5)

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

        # [v2] Steering rate limiter
        max_delta_change = self.steer_rate_max * self._dt
        delta = float(np.clip(delta,
                              self._prev_steer - max_delta_change,
                              self._prev_steer + max_delta_change))
        self._prev_steer = delta

        # 5) speed: waypoint profile
        speed = float(self.waypoints[target_idx].vx_mps)
        if speed < 0.1:
            speed = 1.0

        # [v2] Preview speed — pre-brake before upcoming corners
        preview_indices = [(target_idx + i) % N for i in range(1, self.preview_n + 1)]
        preview_vx = min(float(self.waypoints[i].vx_mps) for i in preview_indices)
        speed = (1.0 - self.preview_blend) * speed + self.preview_blend * preview_vx

        # [v2] Speed low-pass smoothing
        if self._smoothed_speed is None:
            self._smoothed_speed = speed
        self._smoothed_speed = (self.speed_alpha * self._smoothed_speed
                                + (1.0 - self.speed_alpha) * speed)
        speed = self._smoothed_speed

        # [v2] Boost on clear straights
        if abs(kappa_near) < self.straight_kappa and self._is_path_clear(self.clear_dist):
            speed = min(speed * self.speed_boost, self.v_max_boost)

        return delta, speed

    def _is_path_clear(self, dist_threshold):
        if self.scan is None:
            return False
        scan = self.scan
        n = len(scan.ranges)
        i_center = int(round((0.0 - scan.angle_min) / scan.angle_increment))
        i_half   = int(round(math.pi / 6.0 / scan.angle_increment))
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

    def _publish_viz(self, car_pos, yaw, lookahead_r, nearest_wp, cte, in_recovery):
        now = self.get_clock().now().to_msg()

        # ── 1) CTE line: car → nearest waypoint ──────────────────────────────
        # Color: green (small error) → red (large error), threshold = recovery_thresh
        t = min(abs(cte) / self.cte_recovery_thresh, 1.0)
        cte_m = Marker()
        cte_m.header.stamp = now; cte_m.header.frame_id = 'map'
        cte_m.ns = 'pp_cte'; cte_m.id = 0
        cte_m.type = Marker.LINE_STRIP; cte_m.action = Marker.ADD
        cte_m.scale.x = 0.05
        cte_m.color.r = t; cte_m.color.g = 1.0 - t; cte_m.color.b = 0.0; cte_m.color.a = 1.0
        from geometry_msgs.msg import Point
        p0 = Point(); p0.x = car_pos.x;      p0.y = car_pos.y;      p0.z = 0.05
        p1 = Point(); p1.x = nearest_wp.x_m; p1.y = nearest_wp.y_m; p1.z = 0.05
        cte_m.points = [p0, p1]
        self.cte_pub.publish(cte_m)

        # ── 2) Lookahead circle around car ────────────────────────────────────
        circle_m = Marker()
        circle_m.header.stamp = now; circle_m.header.frame_id = 'map'
        circle_m.ns = 'pp_circle'; circle_m.id = 0
        circle_m.type = Marker.LINE_STRIP; circle_m.action = Marker.ADD
        circle_m.scale.x = 0.04
        # Blue in normal mode, orange in recovery
        if in_recovery:
            circle_m.color.r = 1.0; circle_m.color.g = 0.5; circle_m.color.b = 0.0
        else:
            circle_m.color.r = 0.2; circle_m.color.g = 0.6; circle_m.color.b = 1.0
        circle_m.color.a = 0.8
        n_seg = 32
        pts = []
        for i in range(n_seg + 1):
            angle = 2.0 * math.pi * i / n_seg
            pt = Point()
            pt.x = car_pos.x + lookahead_r * math.cos(angle)
            pt.y = car_pos.y + lookahead_r * math.sin(angle)
            pt.z = 0.05
            pts.append(pt)
        circle_m.points = pts
        self.circle_pub.publish(circle_m)

        # ── 3) Status text above car ──────────────────────────────────────────
        txt_m = Marker()
        txt_m.header.stamp = now; txt_m.header.frame_id = 'map'
        txt_m.ns = 'pp_status'; txt_m.id = 0
        txt_m.type = Marker.TEXT_VIEW_FACING; txt_m.action = Marker.ADD
        txt_m.pose.position.x = car_pos.x
        txt_m.pose.position.y = car_pos.y
        txt_m.pose.position.z = 0.6
        txt_m.pose.orientation.w = 1.0
        txt_m.scale.z = 0.25
        if in_recovery:
            txt_m.text = f'RECOVERY\nCTE={cte:.2f}m'
            txt_m.color.r = 1.0; txt_m.color.g = 0.3; txt_m.color.b = 0.0
        else:
            txt_m.text = f'L_f={lookahead_r:.2f}m\nCTE={cte:.2f}m'
            txt_m.color.r = 1.0; txt_m.color.g = 1.0; txt_m.color.b = 1.0
        txt_m.color.a = 1.0
        self.status_pub.publish(txt_m)


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
