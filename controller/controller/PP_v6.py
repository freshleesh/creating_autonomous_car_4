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
    # [v5] 2-stage kappa-based speed limit
    #   Far  (preview_n개): sqrt(a_lat_max * far_factor / kappa) → 부드러운 사전 감속
    #   Near (alat_near_n개): sqrt(a_lat_max / kappa)           → 코너 직전 하드 제한
    'pp_a_lat_max':       2.5,   # 하드 제한 기준 횡가속도 [m/s²]
    'pp_alat_far_factor': 3.0,   # 원거리 소프트 제한 완화 배수 (클수록 늦게 감속)
    'pp_alat_near_n':     15,    # 근거리 하드 제한 윈도우 (15개 ≈ 2.5m)
    # [v6] CTE derivative damping
    'pp_Kd_cte':           0.05,
    # [v6] Behind kappa check: 코너 탈출 직후 부스트 방지
    #   뒤쪽 N개 waypoint에도 코너가 없어야 부스트 허용
    #   직선 중간에서는 뒤도 straight → 영향 없음
    'pp_alat_behind_n':    10,
    # [v6] Target heading feedforward: target waypoint 방향으로 선행 보정
    #   nearest heading은 현재 위치 기준, target heading은 목표 지점 기준
    #   코너 진입 시 target이 이미 코너 안에 있어 미리 꺾기 시작
    'pp_K_heading_target': 0.10,
    # [v6] Speed-adaptive K_heading: 고속일수록 heading 게인 자동 증가
    #   K_heading_eff = K_heading * clip(v / v_ref, 1.0, 2.0)
    #   v_ref 이하: 게인 그대로, v_ref 2배: 최대 2× 증폭
    'pp_v_heading_ref':    3.5,
    # [v4] Speed-proportional steering limit: delta_max = atan(a_lat_steer * L / v²)
    #   25 → 5m/s코너: 0.4rad(풀), 7m/s직선: 0.15rad, 9m/s: 0.09rad
    'pp_a_lat_steer':     25.0,
    # [v4] CTE-proportional speed reduction: spin 방지의 실질 역할
    #   이탈할수록 속도 자동 감쇠: v_out = v / (1 + k * CTE²)
    #   0.0 = 비활성, 0.5 = 보통, 1.5 = 강하게
    'pp_k_v_cte':         0.8,
    # [v3] Global speed scale: multiply ALL waypoint vx by this factor
    #   Trajectory optimizer targets a_lat=6 m/s²; PP needs ~60-70% of that to track
    'pp_v_scale':         0.65,  # 1.0 = use optimizer speed as-is, 0.65 = 35% slower
    # [v6] Startup / low-speed stability. Below v_track_min the adaptive lookahead
    #   collapses to its minimum and the CTE gains slam the wheel → big initial
    #   left-right swing. Below this speed we raise the lookahead floor and fade
    #   in the CTE correction. No effect above v_track_min, so cornering is unchanged.
    'pp_v_track_min':       1.5,   # [m/s] speed at which full control authority is reached
    'pp_startup_lookahead': 1.8,   # [m] lookahead floor at standstill
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
        self.alat_far_factor      = p('pp_alat_far_factor')
        self.alat_near_n          = int(p('pp_alat_near_n'))
        self.Kd_cte               = p('pp_Kd_cte')
        self.alat_behind_n        = int(p('pp_alat_behind_n'))
        self.K_heading_target     = p('pp_K_heading_target')
        self.v_heading_ref        = p('pp_v_heading_ref')
        self.a_lat_steer          = p('pp_a_lat_steer')
        self.k_v_cte              = p('pp_k_v_cte')
        self.v_scale              = p('pp_v_scale')
        self.v_track_min          = p('pp_v_track_min')
        self.startup_lookahead    = p('pp_startup_lookahead')

        self.scan      = None
        self.odom      = None
        self.waypoints = []

        self._prev_steer = 0.0          # [v3] for steering rate limiter
        self._prev_cte   = None         # [v6] None = skip D term on first call
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
            f'PPNode v6 ready | v_scale={self.v_scale:.2f}  a_lat_max={self.a_lat_max:.1f}'
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

        # Corner preview: max |kappa| over the path within lookahead_max ahead.
        # Using kappa_near (curvature at the car) only shrinks the lookahead once
        # the car is ALREADY in the corner. Scanning ahead lets it shrink as we
        # APPROACH the corner, so we don't carry a long (corner-cutting) lookahead
        # into the turn.
        kappa_corner = abs(kappa_near)
        acc_k  = 0.0
        prev_k = nearest_idx
        for step in range(1, N):
            idx = (nearest_idx + step) % N
            acc_k += float(np.hypot(wp_xy[idx, 0] - wp_xy[prev_k, 0],
                                    wp_xy[idx, 1] - wp_xy[prev_k, 1]))
            kappa_corner = max(kappa_corner,
                               abs(float(self.waypoints[idx].kappa_radpm)))
            if acc_k >= self.lookahead_max:
                break
            prev_k = idx

        # Adaptive lookahead — shrinks with CTE (recovery) AND upcoming curvature.
        # kappa_corner > 0 BEFORE the car reaches the corner → L_f shrinks early.
        lookahead = float(np.clip(
            self.lookahead_gain * v / (
                1.0
                + self.cte_gain * abs(cte)
                + self.kappa_lookahead_gain * kappa_corner
            ),
            self.lookahead_min,
            self.lookahead_max,
        ))

        # Startup / low-speed stability: at v≈0 the lookahead above collapses to
        # its minimum (twitchy) and the CTE gains slam the wheel. Below
        # v_track_min, raise the lookahead floor and fade in the CTE correction so
        # the car eases onto the line. low_speed_scale: 0 at standstill → 1 at v_track_min.
        low_speed_scale = float(np.clip(v / max(self.v_track_min, 1e-3), 0.0, 1.0))
        lookahead = max(lookahead, self.startup_lookahead * (1.0 - low_speed_scale))

        # 3) target waypoint
        if not np.any(ahead):
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

        # Walk FORWARD along the path from the nearest waypoint, accumulating
        # arc length, and take the first point at least `lookahead` along the
        # PATH. Selecting by raw straight-line distance (argmin|dist - L|) could
        # snap the target onto a point on the far side of the track that happens
        # to sit ~L away on a tight curve, flinging the lookahead across the
        # track and cutting the corner. Arc-length-forward can never jump across.
        target_idx = nearest_idx
        acc  = 0.0
        prev = nearest_idx
        for step in range(1, N):
            idx = (nearest_idx + step) % N
            acc += float(np.hypot(wp_xy[idx, 0] - wp_xy[prev, 0],
                                  wp_xy[idx, 1] - wp_xy[prev, 1]))
            target_idx = idx
            if acc >= lookahead:
                break
            prev = idx

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
        # [v6 수정] nearest_idx → target_idx 로 통일
        #   PP 기하학도 target_idx 기준 → 같은 점을 보므로 서로 반대로 당기는 문제 해소
        psi_ref     = float(self.waypoints[target_idx].psi_rad)
        heading_err = math.atan2(math.sin(psi_ref - yaw), math.cos(psi_ref - yaw))

        # [v6] Speed-adaptive K_heading
        speed_scale = float(np.clip(v / max(self.v_heading_ref, 0.1), 1.0, 2.0))
        delta += self.K_heading * speed_scale * heading_err + self.Kp_cte * low_speed_scale * cte

        # [v6] Target heading feedforward (target_idx와 동일 기준이 됐으므로 0 권장)
        psi_target_ff   = float(self.waypoints[target_idx].psi_rad)
        heading_err_tgt = math.atan2(math.sin(psi_target_ff - yaw), math.cos(psi_target_ff - yaw))
        delta += self.K_heading_target * heading_err_tgt

        # [v6] CTE derivative damping
        #   복귀 중(dcte/dt < 0): delta 감소 → 오버슈트 방지
        #   초기 호출(_prev_cte=None)과 waypoint 갱신 시 스파이크 방지
        if self._prev_cte is None:
            dcte_dt = 0.0
        else:
            raw_dcte = (cte - self._prev_cte) / self._dt
            dcte_dt  = float(np.clip(raw_dcte, -10.0, 10.0))  # ±10 m/s 클램프
        self._prev_cte = cte
        delta += self.Kd_cte * low_speed_scale * dcte_dt

        delta = max(-self.max_steer, min(self.max_steer, delta))

        # [v4] Speed-proportional steering limit
        #   a_lat_steer=25 → 5m/s(코너): 0.4rad(풀), 7m/s(직선): 0.15rad, 9m/s: 0.09rad
        #   코너(kappa↑)에서는 완화, 고속 직선에서만 의미있게 제한
        kappa_factor = 1.0 + 5.0 * abs(kappa_near)
        delta_max_v = float(np.clip(
            math.atan(self.a_lat_steer * self.wheelbase * kappa_factor / max(v * v, 0.1)),
            0.05,
            self.max_steer,
        ))
        delta = float(np.clip(delta, -delta_max_v, delta_max_v))

        # [v3] Steering rate limiter
        max_chg = self.steer_rate_max * self._dt
        delta = float(np.clip(delta,
                               self._prev_steer - max_chg,
                               self._prev_steer + max_chg))
        self._prev_steer = delta

        # 5) speed — 2-stage kappa-based limit
        speed = float(self.waypoints[target_idx].vx_mps) * self.v_scale
        if speed < 0.1:
            speed = 1.0

        # Stage 1 — FAR soft cap (차량 실제 위치 nearest_idx 기준)
        #   target_idx 기준이면 코너에서 target이 이미 직선에 있어 cap이 안 걸림
        #   nearest_idx 기준으로 하면 차가 직선에 있을 때 다음 코너를 정확히 봄
        preview_indices = [(nearest_idx + i) % N for i in range(1, self.preview_n + 1)]
        far_kappas = [abs(float(self.waypoints[i].kappa_radpm)) for i in preview_indices]
        max_kappa_far = max(far_kappas) if far_kappas else 0.0
        if max_kappa_far > 1e-6:
            v_far_cap = math.sqrt(self.a_lat_max * self.alat_far_factor / max_kappa_far)
            speed = min(speed, v_far_cap)

        # Stage 2 — NEAR hard cap (nearest_idx 기준)
        near_indices = [(nearest_idx + i) % N for i in range(1, self.alat_near_n + 1)]
        near_kappas = [abs(float(self.waypoints[i].kappa_radpm)) for i in near_indices]
        max_kappa_near = max(near_kappas) if near_kappas else 0.0
        if max_kappa_near > 1e-6:
            v_near_cap = math.sqrt(self.a_lat_max / max_kappa_near)
            speed = min(speed, v_near_cap)

        # [v4] CTE-proportional speed reduction
        speed = speed / (1.0 + self.k_v_cte * cte * cte)

        # [v5] Boost — 세 가지 조건 모두 만족해야 부스트
        #   1) kappa_near    : 지금 직선인가
        #   2) max_kappa_near: 앞 4.5m도 직선인가
        #   3) kappa_behind  : 뒤 10개(1.7m)도 직선인가 → 코너 탈출 직후 부스트 방지
        #      (직선 중간에서는 뒤도 straight라 영향 없음)
        kappa_max_behind = max(
            abs(float(self.waypoints[(nearest_idx - i) % N].kappa_radpm))
            for i in range(1, self.alat_behind_n + 1)
        )
        if (abs(kappa_near) < self.straight_kappa
                and max_kappa_near < self.straight_kappa
                and kappa_max_behind < self.straight_kappa):
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
