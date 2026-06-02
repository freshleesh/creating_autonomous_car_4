import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from f110_msgs.msg import WpntArray

from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
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
    # Speed-adaptive lookahead: L_f = clip(gain * v / (1 + cte_gain*|CTE|), min, max)
    'pp_lookahead_gain':  0.35,
    'pp_lookahead_min':   0.8,
    'pp_lookahead_max':   2.5,
    # Curvature feedforward
    'pp_ff_gain':         0.5,
    # CTE-adaptive lookahead shrink
    'pp_cte_gain':        2.0,
    # Direct error feedback
    'pp_Kp_cte':          0.3,
    'pp_K_heading':       0.4,
    # [v3] Steering rate limiter: max steering change per second [rad/s]
    'pp_steer_rate_max':  1.5,
    # [v3] Preview speed: look N waypoints ahead, blend min speed in
    'pp_preview_n':       20,
    'pp_preview_blend':   0.6,
    # [v3] Curvature-adaptive lookahead: L_f shrinks at corners even before CTE grows
    #   L_f = clip(gain * v / (1 + cte_gain*|CTE| + kappa_gain*|kappa|), min, max)
    'pp_kappa_lookahead_gain': 1.5,
    # [v3] Lateral acceleration cap: v_max = sqrt(a_lat_max / kappa)
    #   Hard physics limit — prevents corner entry at physically impossible speed
    'pp_a_lat_max':       4.0,   # [m/s²]  lower = slower corners, higher = faster
    # [v3] Global speed scale: multiply ALL waypoint vx by this factor
    #   Trajectory optimizer targets a_lat=6 m/s²; PP needs ~60-70% of that to track
    'pp_v_scale':         0.65,  # 1.0 = use optimizer speed as-is, 0.65 = 35% slower
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
        self.kappa_lookahead_gain = p('pp_kappa_lookahead_gain')
        self.a_lat_max            = p('pp_a_lat_max')
        self.v_scale              = p('pp_v_scale')

        self.scan      = None
        self.odom      = None
        self.waypoints = []

        self._prev_steer = 0.0          # [v3] for steering rate limiter
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
        self.drive_pub        = self.create_publisher(AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.lookahead_pub    = self.create_publisher(Marker,      '/pp/lookahead',        10)
        self.cte_pub          = self.create_publisher(Marker,      '/pp/cte_line',         10)
        self.circle_pub       = self.create_publisher(Marker,      '/pp/lookahead_circle', 10)
        self.status_pub       = self.create_publisher(Marker,      '/pp/status_text',      10)
        self.wp_speed_pub     = self.create_publisher(MarkerArray, '/pp/waypoints_speed',  10)
        self.preview_pub      = self.create_publisher(Marker,      '/pp/preview',          10)
        self.create_timer(self._dt, self._loop)

        self.get_logger().info(
            f'PPNode v3 ready | v_scale={self.v_scale:.2f}  a_lat_max={self.a_lat_max:.1f}'
            f'  steer_rate={self.steer_rate_max:.1f}  L_f=[{self.lookahead_min:.1f},{self.lookahead_max:.1f}]'
        )

    def _scan_cb(self, msg): self.scan = msg
    def _odom_cb(self, msg): self.odom = msg
    def _wp_cb(self, msg):
        self.waypoints = msg.wpnts
        self._publish_waypoint_speed_colormap()

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
        dists   = np.hypot(dx, dy)

        # 2) nearest waypoint for CTE and feedforward
        nearest_idx = int(np.argmin(dists))
        cte         = float(local_y[nearest_idx])
        kappa_near  = float(self.waypoints[nearest_idx].kappa_radpm)

        # Adaptive lookahead — shrinks with CTE (recovery) AND curvature (corner entry)
        # Key: kappa_near > 0 at corners BEFORE the car deviates → L_f shrinks early
        lookahead = float(np.clip(
            self.lookahead_gain * v / (
                1.0
                + self.cte_gain * abs(cte)
                + self.kappa_lookahead_gain * abs(kappa_near)
            ),
            self.lookahead_min,
            self.lookahead_max,
        ))

        # 3) target waypoint
        err = np.abs(dists - lookahead)
        err[~ahead] = np.inf

        if np.all(np.isinf(err)):
            # [v3] No ahead waypoints: steer toward nearest using heading correction
            #      only — skip PP geometry to avoid wrong-direction steering.
            psi_ref     = float(self.waypoints[nearest_idx].psi_rad)
            heading_err = math.atan2(math.sin(psi_ref - yaw),
                                     math.cos(psi_ref - yaw))
            delta = 0.5 * heading_err
            delta = max(-self.max_steer, min(self.max_steer, delta))
            max_chg = self.steer_rate_max * self._dt
            delta = float(np.clip(delta,
                                  self._prev_steer - max_chg,
                                  self._prev_steer + max_chg))
            self._prev_steer = delta
            self.get_logger().warn(
                f'[PP] no ahead waypoints, heading recovery: err={math.degrees(heading_err):.1f}°',
                throttle_duration_sec=0.5)
            return delta, 0.8
        else:
            target_idx = int(np.argmin(err))

        self._publish_lookahead(self.waypoints[target_idx])

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
        psi_ref     = float(self.waypoints[nearest_idx].psi_rad)
        heading_err = math.atan2(math.sin(psi_ref - yaw), math.cos(psi_ref - yaw))
        delta += self.K_heading * heading_err + self.Kp_cte * cte

        delta = max(-self.max_steer, min(self.max_steer, delta))

        # [v3] Steering rate limiter
        max_chg = self.steer_rate_max * self._dt
        delta = float(np.clip(delta,
                               self._prev_steer - max_chg,
                               self._prev_steer + max_chg))
        self._prev_steer = delta

        # 5) speed from waypoint profile
        speed = float(self.waypoints[target_idx].vx_mps) * self.v_scale
        if speed < 0.1:
            speed = 1.0

        # [v3] Preview speed: pre-brake before upcoming slow corners
        preview_indices = [(target_idx + i) % N for i in range(1, self.preview_n + 1)]
        preview_vx = min(float(self.waypoints[i].vx_mps) * self.v_scale for i in preview_indices)
        speed = (1.0 - self.preview_blend) * speed + self.preview_blend * preview_vx

        # [v3] Lateral acceleration cap: v ≤ sqrt(a_lat_max / kappa)
        # Uses the max curvature in the next preview_n waypoints (same window as speed)
        preview_kappas = [abs(float(self.waypoints[i].kappa_radpm)) for i in preview_indices]
        max_kappa = max(preview_kappas) if preview_kappas else 0.0
        if max_kappa > 1e-6:
            v_lat_cap = math.sqrt(self.a_lat_max / max_kappa)
            speed = min(speed, v_lat_cap)

        # Boost on clear straights
        if abs(kappa_near) < self.straight_kappa and self._is_path_clear(self.clear_dist):
            speed = min(speed * self.speed_boost, self.v_max_boost)

        self._publish_viz(p, yaw, lookahead, nearest_idx, target_idx, cte, speed, preview_indices)
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

    def _publish_viz(self, car_pos, yaw, lookahead_r, nearest_idx, target_idx, cte, speed, preview_indices):
        now = self.get_clock().now().to_msg()

        # ── 1) CTE line: car → nearest waypoint (green → red) ────────────────
        cte_ref = 0.6  # distance at which line goes fully red
        t = min(abs(cte) / cte_ref, 1.0)
        cte_m = Marker()
        cte_m.header.stamp = now; cte_m.header.frame_id = 'map'
        cte_m.ns = 'pp_cte'; cte_m.id = 0
        cte_m.type = Marker.LINE_STRIP; cte_m.action = Marker.ADD
        cte_m.scale.x = 0.05
        cte_m.color.r = t; cte_m.color.g = 1.0 - t; cte_m.color.b = 0.0; cte_m.color.a = 1.0
        p0 = Point(); p0.x = car_pos.x; p0.y = car_pos.y; p0.z = 0.05
        nw = self.waypoints[nearest_idx]
        p1 = Point(); p1.x = nw.x_m;   p1.y = nw.y_m;   p1.z = 0.05
        cte_m.points = [p0, p1]
        self.cte_pub.publish(cte_m)

        # ── 2) Lookahead circle ───────────────────────────────────────────────
        circle_m = Marker()
        circle_m.header.stamp = now; circle_m.header.frame_id = 'map'
        circle_m.ns = 'pp_circle'; circle_m.id = 0
        circle_m.type = Marker.LINE_STRIP; circle_m.action = Marker.ADD
        circle_m.scale.x = 0.04
        circle_m.color.r = 0.2; circle_m.color.g = 0.6; circle_m.color.b = 1.0; circle_m.color.a = 0.8
        n_seg = 36
        pts = []
        for i in range(n_seg + 1):
            a = 2.0 * math.pi * i / n_seg
            pt = Point()
            pt.x = car_pos.x + lookahead_r * math.cos(a)
            pt.y = car_pos.y + lookahead_r * math.sin(a)
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
        txt_m.pose.position.z = 0.7
        txt_m.pose.orientation.w = 1.0
        txt_m.scale.z = 0.2
        txt_m.text = f'L_f={lookahead_r:.2f}m  CTE={cte:+.2f}m\nv={speed:.1f}m/s'
        txt_m.color.r = 1.0; txt_m.color.g = 1.0; txt_m.color.b = 1.0; txt_m.color.a = 1.0
        self.status_pub.publish(txt_m)

        # ── 4) Preview highlight (next N waypoints used for speed decision) ───
        prev_m = Marker()
        prev_m.header.stamp = now; prev_m.header.frame_id = 'map'
        prev_m.ns = 'pp_preview'; prev_m.id = 0
        prev_m.type = Marker.SPHERE_LIST; prev_m.action = Marker.ADD
        prev_m.scale.x = prev_m.scale.y = prev_m.scale.z = 0.1
        prev_m.color.a = 1.0
        for i, idx in enumerate(preview_indices):
            wp = self.waypoints[idx]
            pt = Point(); pt.x = wp.x_m; pt.y = wp.y_m; pt.z = 0.08
            prev_m.points.append(pt)
            # color: yellow at start, orange at end
            frac = i / max(len(preview_indices) - 1, 1)
            c = ColorRGBA()
            c.r = 1.0; c.g = 0.8 * (1.0 - frac * 0.6); c.b = 0.0; c.a = 0.9
            prev_m.colors.append(c)
        self.preview_pub.publish(prev_m)

    def _publish_waypoint_speed_colormap(self):
        """waypoint 전체를 속도에 따라 파랑→빨강으로 색칠. 수신 시 1회 발행."""
        if not self.waypoints:
            return
        vx_all  = [float(w.vx_mps) for w in self.waypoints]
        vx_min_ = min(vx_all)
        vx_range = max(max(vx_all) - vx_min_, 0.1)

        ma = MarkerArray()
        m = Marker()
        m.header.stamp    = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns = 'pp_wp_speed'; m.id = 0
        m.type = Marker.SPHERE_LIST; m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.08
        m.color.a = 1.0  # required even for per-point colors

        for w, vx in zip(self.waypoints, vx_all):
            pt = Point(); pt.x = w.x_m; pt.y = w.y_m; pt.z = 0.03
            m.points.append(pt)
            t = (vx - vx_min_) / vx_range  # 0=slow(blue), 1=fast(red)
            c = ColorRGBA()
            c.r = t;       c.g = 0.3 * (1.0 - abs(t - 0.5) * 2); c.b = 1.0 - t; c.a = 0.9
            m.colors.append(c)

        ma.markers.append(m)
        self.wp_speed_pub.publish(ma)


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
