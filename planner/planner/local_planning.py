#!/usr/bin/env python3

#로컬플래닝 0606

"""
local_planning.py - Standalone mode-selectable local planner.

Single file: perception + Frenet + planner. PP receives this node's
/local_waypoints via launch-level topic remap (PP.py unchanged).

Pipeline:
    /scan, /vesc/odom, /global_waypoints                          INPUT
        1. perception  : cluster -> l_shape_fitting -> tracking
        2. Frenet      : raceline cubic spline + to_frenet/to_cartesian
        3. mode branch :
             free          - raceline forward window
             trailing      - raceline + trailing speed cap
             spline_avoid  - left/right cubic spline candidates,
                             fall back to trailing if both infeasible
        4. publish WpntArray
    /local_waypoints                                              OUTPUT

Run:
    /usr/bin/python3 local_planning.py --ros-args -p mode:=spline_avoid
"""

import math

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from f110_msgs.msg import Wpnt, WpntArray


# ===========================================================================
#  PERCEPTION - Geometry helpers  (provided)
# ===========================================================================

def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def scan_to_xy(ranges: np.ndarray, angle_min: float, angle_inc: float):
    # --- tunable parameters ---
    r_min = 0.05    # [m] ignore returns closer than this
    r_max = 10.0    # [m] ignore returns farther than this

    n = ranges.shape[0]
    angles = angle_min + np.arange(n) * angle_inc
    valid = np.isfinite(ranges) & (ranges >= r_min) & (ranges <= r_max)
    x = np.where(valid, ranges * np.cos(angles), np.nan)
    y = np.where(valid, ranges * np.sin(angles), np.nan)
    return x, y
 
# ===========================================================================
#  PERCEPTION  (Using your code from perception_assignment.py)
# ===========================================================================


def cluster(x: np.ndarray, y: np.ndarray, angle_inc: float):

    
    # --- tunable parameters ---
    lambda_rad = math.radians(30.0)
    sigma      = 0.3   # 클수록 멀리떨어진 점들이 더 클러스터링 잘되게
    min_points = 9      # 클러스터링 충족하는 포인트 수 (벽 반사 단편 제거)


    use_adaptive = angle_inc > 1e-9
    if use_adaptive:
        denom = math.sin(lambda_rad - angle_inc)
        if abs(denom) < 1e-6:
            denom = 1e-6 if denom >= 0 else -1e-6

    clusters = []
    current  = []
    prev     = None

    n = len(x)
    for i in range(n):
        xi, yi = float(x[i]), float(y[i])

        if not (math.isfinite(xi) and math.isfinite(yi)):
            if len(current) >= min_points:
                clusters.append(current)
            current = []
            prev    = None
            continue

        if prev is None:
            current.append((xi, yi))
            prev = (xi, yi)
            continue

        if use_adaptive:
            r     = math.hypot(xi, yi)
            d_max = (r * math.sin(angle_inc) / denom + 3.0 * sigma) / 2.0
        else:
            d_max = 0.3

        jump = math.hypot(xi - prev[0], yi - prev[1])

        if jump > d_max:
            if len(current) >= min_points:
                clusters.append(current)
            current = [(xi, yi)]
        else:
            current.append((xi, yi))
        prev = (xi, yi)

    if len(current) >= min_points:
        clusters.append(current)

    return clusters











def l_shape_fitting(clusters):
    # --- tunable parameters ---
    max_obs_size = 1.3       # 장애물 길이 범위
    min_size     = 0.15       
    min_edge     = 0.01
    n_angles     = 90

    CAR_LENGTH = 0.50
    CAR_WIDTH  = 0.35

    thetas        = np.linspace(0.0, np.pi / 2 - np.pi / 180, n_angles)
    cos_t, sin_t  = np.cos(thetas), np.sin(thetas)
    obstacles     = []

    for cl in clusters:
        pts = np.array(cl, dtype=float)
        if pts.shape[0] < 3:
            continue

        centroid     = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        cov          = np.cov(pts_centered, rowvar=False)
        try:
            eigenvalues, _ = np.linalg.eigh(cov)
            if max(eigenvalues) / (min(eigenvalues) + 1e-6) > 80.0:
                continue
        except np.linalg.LinAlgError:
            continue

        a = pts_centered[:, 0:1] * cos_t + pts_centered[:, 1:2] * sin_t
        b = -pts_centered[:, 0:1] * sin_t + pts_centered[:, 1:2] * cos_t

        a_min, a_max = a.min(axis=0), a.max(axis=0)
        b_min, b_max = b.min(axis=0), b.max(axis=0)

        da    = np.minimum(a - a_min, a_max - a)
        db    = np.minimum(b - b_min, b_max - b)
        d     = np.minimum(da, db)
        score = np.sum(1.0 / np.maximum(d, min_edge), axis=0)

        w_cand = a_max - a_min
        h_cand = b_max - b_min

        size_penalty = (
            np.exp(-((w_cand - 0.45)**2) / 0.25) * np.exp(-((h_cand - 0.35)**2) / 0.25) +
            np.exp(-((w_cand - 0.35)**2) / 0.25) * np.exp(-((h_cand - 0.45)**2) / 0.25)
        )
        k = int(np.argmax(score * size_penalty))

        c = float(cos_t[k]); s = float(sin_t[k])
        w = float(a_max[k] - a_min[k])
        h = float(b_max[k] - b_min[k])

        if max(w, h) > max_obs_size:
            continue

        if w > h:
            w, h = CAR_LENGTH, CAR_WIDTH
        else:
            w, h = CAR_WIDTH, CAR_LENGTH

        ca = 0.5 * (a_max[k] + a_min[k])
        cb = 0.5 * (b_max[k] + b_min[k])
        cx = float(ca * c - cb * s + centroid[0])
        cy = float(ca * s + cb * c + centroid[1])

        obstacles.append((cx, cy, w, h))

    return obstacles








def tracking(obstacles, track, dt: float, ego, max_misses: int = 60):
    # --- tunable parameters ---
    opp_max_lat     = 5       # 좌우 트래킹 범위
    Q_scale         = 0.5     # 칼만필터 노이즈 공분산
    R_scale         = 0.5     # 측정 노이즈 가중치
    assoc_threshold = 1.5     # 동일한 물체로 인식하는 거리 기준 (코너 예측 오차 대응)
    lidar_to_base_x = 0.5     # 라이다 기준 좌표 거리

    ex, ey, eyaw = ego

    meas = None
    best_dist = float('inf')
    for (cx_l, cy_l, w, h) in obstacles:
        if not (0.0 < cx_l < 15.0 and abs(cy_l) <= opp_max_lat):
            continue
        bx = cx_l + lidar_to_base_x
        by = cy_l
        gx = ex + bx * math.cos(eyaw) - by * math.sin(eyaw)
        gy = ey + bx * math.sin(eyaw) + by * math.cos(eyaw)

        dist = math.hypot(gx - ex, gy - ey)
        if dist < best_dist:
            best_dist = dist
            meas = (gx, gy)

    if track is None:
        if meas is None:
            return None
        state = np.array([meas[0], meas[1], 0.0, 0.0])
        P     = np.eye(4)
        return (state, P, 0, 1)

    state, P, misses, hits = track

    # 3. Predict
    F = np.array([
        [1.0, 0.0,  dt, 0.0],
        [0.0, 1.0, 0.0,  dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    Q = np.diag([Q_scale, Q_scale, Q_scale * 0.1, Q_scale * 0.1])
    state = F @ state
    P     = F @ P @ F.T + Q

    # 4. Update / coast
    if meas is not None:
        pred_pos = state[:2]
        if math.hypot(meas[0] - pred_pos[0], meas[1] - pred_pos[1]) <= assoc_threshold:
            H = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ])
            R     = np.eye(2) * R_scale
            z     = np.array([meas[0], meas[1]])
            innov = z - H @ state
            S     = H @ P @ H.T + R
            K     = P @ H.T @ np.linalg.inv(S)
            state = state + K @ innov
            P     = (np.eye(4) - K @ H) @ P
            misses = 0
            hits  += 1
        else:
            # 측정이 gate 밖 → miss(코스팅) — reset 금지 (벽으로 점프 방지)
            misses += 1
            state[2] *= 0.85
            state[3] *= 0.85
    else:
        misses += 1
        state[2] *= 0.85
        state[3] *= 0.85

    if misses > max_misses:
        return None

    return (state, P, misses, hits)















def trailing(track, ego, ego_v) -> float:
    """Speed command from the MAP-frame opponent track + ego pose & speed.

    `track` = (state[x,y,vx,vy], P, misses, hits) in the MAP frame, or None.
    `ego`   = (ex, ey, eyaw) ;  `ego_v` = ego forward speed [m/s].
    True PD on the gap: kp on the gap error, kd on the *closing* speed.
    """
    # --- tunable parameters ---
    base_speed    = 8.0   # [m/s] free-running race speed
    desired_gap   = 1.0   # [m] PD 목표 거리
    detect_range  = 6.0   # [m] 이 거리부터 PD 감속 시작
    stop_gap      = 3.0   # [m] 이 거리 이내 → 완전 정지
    kp            = 3.0   # P gain: gap 오차 1m당 속도 보정
    kd            = 4.0   # D gain: closing speed 1m/s당 보정
    max_speed     = 8.0   # [m/s] absolute speed cap
    full_stop     = 0.2   # [m] 최근접 완전 정지
    emergency_stop = 0.5  # [m] 최근접 최저 속도

    # 1. No opponent in view -> race at full speed.
    if track is None:
        return base_speed

    ox, oy, ovx, ovy = track[0]
    ex, ey, eyaw = ego

    # 2. Project the opponent's position and velocity onto the ego heading.
    c, s = math.cos(eyaw), math.sin(eyaw)
    opp_dist =  c * (ox - ex) + s * (oy - ey)   # forward gap [m]
    opp_vx   =  c * ovx       + s * ovy         # opp. forward speed [m/s]
    closing  =  opp_vx - ego_v                  # d(gap)/dt; >0 = pulling away

    # 3. Opponent behind us or out of detect range -> race at full speed.
    if opp_dist < 0.0 or opp_dist > detect_range:
        return base_speed

    # 4. stop_gap 이내 → 완전 정지
    if opp_dist <= stop_gap:
        return 0.0

    # 5. 최근접 안전 임계값
    if opp_dist <= full_stop:
        return 0.0
    if opp_dist <= emergency_stop:
        return 0.5

    # 6. PD speed
    speed = base_speed + kp * (opp_dist - desired_gap) + kd * closing
    speed = max(0.0, min(speed, base_speed, max_speed))

    # 7. 거리 비례 속도 상한: detect_range→stop_gap 구간에서 선형으로 0까지 감속
    ramp = (opp_dist - stop_gap) / (detect_range - stop_gap)  # 1.0(먼) → 0.0(가까운)
    v_ramp = base_speed * ramp
    speed = min(speed, v_ramp)

    return max(0.0, speed)




# ===========================================================================
#  GEOMETRY HELPER (provided)
# ===========================================================================
def geom_psi_kappa(x: np.ndarray, y: np.ndarray):
    """Heading & signed curvature of a non-closed (x, y) sequence."""
    n = len(x)
    psi = np.zeros(n)
    kappa = np.zeros(n)
    for i in range(n):
        if i == 0:
            dx = x[1] - x[0]; dy = y[1] - y[0]
        elif i == n - 1:
            dx = x[-1] - x[-2]; dy = y[-1] - y[-2]
        else:
            dx = (x[i + 1] - x[i - 1]) * 0.5
            dy = (y[i + 1] - y[i - 1]) * 0.5
        psi[i] = math.atan2(dy, dx)
        if 0 < i < n - 1:
            ddx = x[i + 1] - 2 * x[i] + x[i - 1]
            ddy = y[i + 1] - 2 * y[i] + y[i - 1]
            denom = (dx * dx + dy * dy) ** 1.5
            kappa[i] = (dx * ddy - dy * ddx) / max(denom, 1e-9)
    return psi, kappa


# ===========================================================================
#  ROS 2 NODE
# ===========================================================================

class LocalPlanning(Node):

    def __init__(self):
        super().__init__('local_planning')

        gp = lambda name, val: self.declare_parameter(name, val).value

        # ---- mode ('free' | 'trailing' | 'spline_avoid') --------------------
        self.mode = str(gp('mode', 'spline_avoid'))

        # ---- topics ---------------------------------------------------------
        self.scan_topic   = str(gp('scan_topic',   '/scan'))
        self.odom_topic   = str(gp('odom_topic',   '/vesc/odom'))
        self.global_topic = str(gp('global_topic', '/global_waypoints'))
        self.local_topic  = str(gp('local_topic',  '/local_waypoints'))

        # ---- Frenet slice ---------------------------------------------------
        self.local_horizon = float(gp('local_horizon', 5.0))    # [m]
        self.ds_step       = float(gp('ds_step',        0.25))  # [m]
        # ego_d -> target cosine blend distance, ~ 2 * pp_lookahead
        self.s_blend       = float(gp('s_blend',        3.0))   # [m]

        # ---- avoidance (spline_avoid) ---------------------------------------
        self.d_safe        = float(gp('d_safe',       0.3))     # [m] lateral offset
        self.s_in          = float(gp('s_in',         2.0))     # [m] min approach gap
        self.s_out         = float(gp('s_out',        2.5))     # [m] peak -> raceline
        self.trigger_range = float(gp('trigger_range', 1.5))    # [m] forward trigger range
        self.margin        = float(gp('margin',       0.1))     # [m] wall/obstacle margin
        self.obs_radius    = float(gp('obs_radius',   0.4))     # [m] obstacle inflate
        self.track_half_w  = float(gp('track_half_w', 0.8))     # [m] fallback half-width
        self.a_lat_max     = float(gp('a_lat_max',    6.0))     # [m/s^2] lat accel cap
        self.vx_scale_avoid = float(gp('vx_scale_avoid', 0.5))  # avoidance vx multiplier

        # ---- wall clamping (per-sample clamp + PCHIP refit) -----------------
        self.clamp_to_walls = bool(gp('clamp_to_walls', True))
        self.clamp_buffer   = float(gp('clamp_buffer',  0.05))     # [m]

        # ---- avoidance shape ------------------------------------------------
        self.s_hold = float(gp('s_hold', 1.5))  # [m] 가까운 구간 경로 유지 거리

        # ---- Frenet state ---------------------------------------------------
        self._sx = None
        self._sy = None
        self._vx_pchip = None
        self._dl_pchip = None
        self._dr_pchip = None
        self._s_samples = None
        self._x_samples = None
        self._y_samples = None
        self.s_total = 0.0

        # ---- perception / pose state ----------------------------------------
        self.track = None
        self._front_stop = False       # -10°~+10°, 15cm 이내 장애물 → 정지
        self.last_scan_t = None
        self._occ_map    = None
        self._map_array  = None
        self._map_res    = None
        self._map_ox     = None
        self._map_oy     = None
        self._map_w      = None
        self._map_h      = None
        self.ex = self.ey = self.eyaw = 0.0
        self.ev = 0.0
        self.have_pose = False
        self.ego_s = 0.0
        self.ego_d = 0.0

        # ---- avoidance commit (hysteresis) ----------------------------------
        # Hold the same spline until ego passes s_d (raceline rejoin point).
        self._avoid_state = None
        self._avoid_committed_time = None   # 정적 회피 시작 시각 (3초 hold)

        # ---- ROS interfaces -------------------------------------------------
        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(WpntArray,    self.global_topic, self._global_wp_cb, latched)
        self.create_subscription(Odometry,     self.odom_topic,   self._odom_cb, 10)
        self.create_subscription(LaserScan,    self.scan_topic,   self._scan_cb, 10)
        self.create_subscription(OccupancyGrid, '/map',           self._map_cb, latched)

        self.local_pub  = self.create_publisher(WpntArray,   self.local_topic, latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/local_waypoints/markers', 10)
        self.cand_pub   = self.create_publisher(MarkerArray, '/local_planning/candidates', 5)
        self.det_pub    = self.create_publisher(MarkerArray, '/local_planning/detections', 5)
        self.trk_pub    = self.create_publisher(MarkerArray, '/local_planning/tracking', 5)

        self.get_logger().info(
            f'local_planning up | mode={self.mode} | horizon={self.local_horizon} m | '
            f'global={self.global_topic} -> local={self.local_topic}')

    # ================================================================== #
    # ROS callbacks
    # ================================================================== #
    def _map_cb(self, msg):
        self._occ_map = msg
        info = msg.info
        self._map_array  = np.array(msg.data, dtype=np.int8).reshape(info.height, info.width)
        self._map_res    = info.resolution
        self._map_ox     = info.origin.position.x
        self._map_oy     = info.origin.position.y
        self._map_w      = info.width
        self._map_h      = info.height

    def _is_on_static_map(self, gx, gy, w=0.5, h=0.35, threshold=50):
        """중심점 3×3 셀 중 2개 이상 occupied → 벽 위 클러스터로 판정.

        벽 가까이 있는 실제 장애물 오탐을 막기 위해 엄격한 조건 유지.
        (5×5 any-occupied는 벽 0.10m 이내 실제 장애물도 걸러냄)
        """
        if self._map_array is None:
            return False
        col = int((gx - self._map_ox) / self._map_res)
        row = int((gy - self._map_oy) / self._map_res)
        occupied = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = row + dr, col + dc
                if 0 <= r < self._map_h and 0 <= c < self._map_w:
                    if self._map_array[r, c] >= threshold:
                        occupied += 1
        return occupied >= 2

    def _remove_wall_scan_points(self, x, y, threshold=50, radius=1):
        """맵의 occupied 셀 주변 radius 셀 이내 스캔 포인트를 NaN으로 제거."""
        if self._map_array is None:
            return x, y
        lidar_to_base_x = 0.27
        cyaw = math.cos(self.eyaw)
        syaw = math.sin(self.eyaw)
        valid = np.isfinite(x) & np.isfinite(y)
        bx = np.where(valid, x + lidar_to_base_x, 0.0)
        by = np.where(valid, y, 0.0)
        gx = self.ex + cyaw * bx - syaw * by
        gy = self.ey + syaw * bx + cyaw * by
        cols = ((gx - self._map_ox) / self._map_res).astype(int)
        rows = ((gy - self._map_oy) / self._map_res).astype(int)
        in_bounds = valid & (cols >= 0) & (cols < self._map_w) & (rows >= 0) & (rows < self._map_h)
        on_wall = np.zeros(len(x), dtype=bool)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r2 = np.clip(rows + dr, 0, self._map_h - 1)
                c2 = np.clip(cols + dc, 0, self._map_w - 1)
                on_wall |= in_bounds & (self._map_array[r2, c2] >= threshold)
        x = x.copy()
        y = y.copy()
        x[on_wall] = float('nan')
        y[on_wall] = float('nan')
        return x, y

    def _global_wp_cb(self, msg):
        self._build_frenet_spline(list(msg.wpnts))

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.ex, self.ey = p.x, p.y
        self.eyaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.ev   = msg.twist.twist.linear.x
        self.have_pose = True

    def _scan_cb(self, msg):
        if not self.have_pose or self._sx is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.05 if self.last_scan_t is None else max(now - self.last_scan_t, 1e-3)
        self.last_scan_t = now

        # 1) perception
        ranges = np.asarray(msg.ranges, dtype=float)

        # 전방 -10°~+10°, 15cm 이내 장애물 → 정지 플래그
        angles_all = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        front_mask = (
            (angles_all >= math.radians(-10)) & (angles_all <= math.radians(10))
            & np.isfinite(ranges) & (ranges > 0.0)
        )
        self._front_stop = bool(np.any(front_mask & (ranges < 0.15)))

        x, y = scan_to_xy(ranges, msg.angle_min, msg.angle_increment)
        x, y = self._remove_wall_scan_points(x, y, radius=1)  # 벽 포인트 제거 후 클러스터링
        clusters_xy = cluster(x, y, msg.angle_increment)
        obstacles = l_shape_fitting(clusters_xy)

        # 장애물 필터링: 맵 기반 + Frenet 트랙 범위
        lidar_to_base_x = 0.27
        dynamic_obstacles = []
        for (cx_l, cy_l, w, h) in obstacles:
            bx = cx_l + lidar_to_base_x
            by = cy_l
            gx = self.ex + bx * math.cos(self.eyaw) - by * math.sin(self.eyaw)
            gy = self.ey + bx * math.sin(self.eyaw) + by * math.cos(self.eyaw)
            # 맵 필터
            if self._is_on_static_map(gx, gy, w, h):
                continue

            # Frenet 필터: 트랙 범위 밖이면 벽으로 간주
            if self._sx is not None:
                s_obs, d_obs = self.to_frenet(gx, gy)
                dl = self._dl_at(s_obs)
                dr = self._dr_at(s_obs)
                wall_buf = 0.1
                if d_obs > dl - wall_buf or d_obs < -dr + wall_buf:
                    continue
            dynamic_obstacles.append((cx_l, cy_l, w, h))
        obstacles = dynamic_obstacles

        ego = (self.ex, self.ey, self.eyaw)
        self.track = tracking(obstacles, self.track, dt, ego)

        # 1.5) Publish RViz Markers for Detection and Tracking
        self._publish_detection_markers(obstacles)
        self._publish_tracking_markers()

        # 2) ego frenet
        self.ego_s, self.ego_d = self.to_frenet(self.ex, self.ey)

        # 3) mode branch -> WpntArray
        if self.mode == 'free':
            out, used_mode = self._build_passthrough(), 'free'
        elif self.mode == 'trailing':
            out, used_mode = self._build_trailing(), 'trailing'
        elif self.mode == 'spline_avoid':
            out, used_mode = self._build_spline_avoid_or_fallback()
        else:
            self.get_logger().warn(f"unknown mode '{self.mode}' -> free")
            out, used_mode = self._build_passthrough(), 'free'

        self.local_pub.publish(out)
        self._publish_local_markers(out, used_mode)

    def _publish_detection_markers(self, obstacles):
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        lidar_to_base_x = 0.27
        stamp = self.get_clock().now().to_msg()

        for i, (cx_l, cy_l, w, h) in enumerate(obstacles):
            bx = cx_l + lidar_to_base_x
            by = cy_l
            gx = self.ex + bx * math.cos(self.eyaw) - by * math.sin(self.eyaw)
            gy = self.ey + bx * math.sin(self.eyaw) + by * math.cos(self.eyaw)

            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'detections'
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = gx
            m.pose.position.y = gy
            m.pose.position.z = 0.2
            m.pose.orientation.w = 1.0
            m.scale.x = w
            m.scale.y = h
            m.scale.z = 0.4
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 0.0, 0.5
            ma.markers.append(m)

        self.det_pub.publish(ma)

    def _publish_tracking_markers(self):
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        if self.track is not None:
            state, P, misses, hits = self.track
            gx, gy, vx, vy = state
            stamp = self.get_clock().now().to_msg()

            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'tracking_pos'
            m.id = 0
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(gx)
            m.pose.position.y = float(gy)
            m.pose.position.z = 0.2
            m.pose.orientation.w = 1.0
            m.scale.x = 0.5
            m.scale.y = 0.5
            m.scale.z = 0.5
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 1.0, 0.8
            ma.markers.append(m)

            speed = math.hypot(vx, vy)
            if speed > 0.02:
                # 방향 벡터를 고정 길이(1.0m)로 정규화해서 항상 명확하게 표시
                arrow_len = max(speed * 3.0, 0.5)  # 최소 0.5m, 속도 3배 스케일
                nx = vx / speed; ny = vy / speed
                arr = Marker()
                arr.header.frame_id = 'map'
                arr.header.stamp = stamp
                arr.ns = 'tracking_vel'
                arr.id = 1
                arr.type = Marker.ARROW
                arr.action = Marker.ADD
                p1 = Point()
                p1.x, p1.y, p1.z = float(gx), float(gy), 0.3
                p2 = Point()
                p2.x = float(gx + nx * arrow_len)
                p2.y = float(gy + ny * arrow_len)
                p2.z = 0.3
                arr.points = [p1, p2]
                arr.scale.x = 0.08   # shaft 굵기
                arr.scale.y = 0.20   # head 크기
                arr.scale.z = 0.0
                arr.color.r, arr.color.g, arr.color.b, arr.color.a = 1.0, 0.4, 0.7, 1.0
                ma.markers.append(arr)

        self.trk_pub.publish(ma)

    # ================================================================== #
    # Frenet spline
    # ================================================================== #
    def _build_frenet_spline(self, wpnts):
        if len(wpnts) < 4:
            return
        x  = np.array([w.x_m     for w in wpnts], dtype=float)
        y  = np.array([w.y_m     for w in wpnts], dtype=float)
        vx = np.array([w.vx_mps  for w in wpnts], dtype=float)
        dl = np.array([w.d_left  for w in wpnts], dtype=float)
        dr = np.array([w.d_right for w in wpnts], dtype=float)

        if abs(x[0] - x[-1]) > 1e-6 or abs(y[0] - y[-1]) > 1e-6:
            x  = np.append(x,  x[0])
            y  = np.append(y,  y[0])
            vx = np.append(vx, vx[0])
            dl = np.append(dl, dl[0])
            dr = np.append(dr, dr[0])

        ds = np.hypot(np.diff(x), np.diff(y))
        s  = np.concatenate(([0.0], np.cumsum(ds)))

        keep = np.concatenate(([True], np.diff(s) > 1e-9))
        s = s[keep]; x = x[keep]; y = y[keep]
        vx = vx[keep]; dl = dl[keep]; dr = dr[keep]

        # fallback if d_left / d_right are all zero in the CSV
        if np.all(dl < 1e-3):
            dl = np.full_like(dl, self.track_half_w)
        if np.all(dr < 1e-3):
            dr = np.full_like(dr, self.track_half_w)

        self._sx = CubicSpline(s, x, bc_type='periodic')
        self._sy = CubicSpline(s, y, bc_type='periodic')
        self._vx_pchip = PchipInterpolator(s, vx, extrapolate=False)
        self._dl_pchip = PchipInterpolator(s, dl, extrapolate=False)
        self._dr_pchip = PchipInterpolator(s, dr, extrapolate=False)

        self._s_samples = s[:-1]
        self._x_samples = x[:-1]
        self._y_samples = y[:-1]
        self.s_total = float(s[-1])
        self.get_logger().info(
            f'frenet spline built (s_total={self.s_total:.3f} m, N={len(self._s_samples)})')

    def _psi_kappa_at(self, s):
        s = s % self.s_total
        dx  = float(self._sx(s, 1)); dy  = float(self._sy(s, 1))
        ddx = float(self._sx(s, 2)); ddy = float(self._sy(s, 2))
        psi = math.atan2(dy, dx)
        denom = (dx * dx + dy * dy) ** 1.5
        kappa = (dx * ddy - dy * ddx) / denom if denom > 1e-12 else 0.0
        return psi, kappa

    def _vx_at(self, s):
        return float(self._vx_pchip(s % self.s_total))

    def _dl_at(self, s):
        return float(self._dl_pchip(s % self.s_total))

    def _dr_at(self, s):
        return float(self._dr_pchip(s % self.s_total))

    def to_frenet(self, x, y, n_newton=5):
        if self._sx is None:
            return 0.0, 0.0
        i = int(np.argmin((self._x_samples - x) ** 2 + (self._y_samples - y) ** 2))
        s = float(self._s_samples[i])
        for _ in range(n_newton):
            rx = float(self._sx(s)) - x
            ry = float(self._sy(s)) - y
            dx = float(self._sx(s, 1)); dy = float(self._sy(s, 1))
            ddx = float(self._sx(s, 2)); ddy = float(self._sy(s, 2))
            g  = rx * dx + ry * dy
            gp = dx * dx + dy * dy + rx * ddx + ry * ddy
            if abs(gp) < 1e-12:
                break
            s = (s - g / gp) % self.s_total
        dx = float(self._sx(s, 1)); dy = float(self._sy(s, 1))
        nrm = math.hypot(dx, dy)
        if nrm < 1e-12:
            return s, 0.0
        nx, ny = -dy / nrm, dx / nrm
        d = (x - float(self._sx(s))) * nx + (y - float(self._sy(s))) * ny
        return s, d

    def to_cartesian(self, s, d):
        if self._sx is None:
            return 0.0, 0.0
        s = s % self.s_total
        x0 = float(self._sx(s)); y0 = float(self._sy(s))
        dx = float(self._sx(s, 1)); dy = float(self._sy(s, 1))
        nrm = math.hypot(dx, dy)
        if nrm < 1e-12:
            return x0, y0
        nx, ny = -dy / nrm, dx / nrm
        return x0 + d * nx, y0 + d * ny

    # ================================================================== #
    # Builders
    # ================================================================== #
    def _empty_header(self):
        out = WpntArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'
        return out

    def _n_pts(self):
        return max(2, int(self.local_horizon / self.ds_step) + 1)

    def _make_local_wpnts(self, target_fn, v_cap=None, use_curvature_cap=False,
                          vx_scale=1.0, use_blend=True):
        """Unified builder: target d(s) + (optional) ego cosine blend + vx.

        With use_blend=True (default, raceline/trailing):
            d(s) = target_fn(s) + (ego_d - target_fn(ego_s)) * cos_alpha(s_off)
        With use_blend=False (spline_avoid): d(s) = target_fn(s) — the
        precomputed avoidance spline is published as-is, so what PP sees
        matches the candidate visualization.

        vx = min( raceline PCHIP vx,
                  v_cap (if given, scalar from trailing),
                  sqrt(a_lat_max / |kappa|) (if use_curvature_cap) ).
        """
        n = self._n_pts()
        if use_blend:
            target_at_ego = float(target_fn(self.ego_s))
            delta = self.ego_d - target_at_ego
        else:
            delta = 0.0

        # (s, d) sequence
        s_arr = np.empty(n)
        d_arr = np.empty(n)
        for k in range(n):
            s = (self.ego_s + k * self.ds_step) % self.s_total
            d_target = float(target_fn(s))
            if not use_blend:
                d = d_target
            else:
                s_off = k * self.ds_step
                if s_off >= self.s_blend:
                    d = d_target
                else:
                    alpha = 0.5 * (1.0 + math.cos(math.pi * s_off / self.s_blend))
                    d = d_target + delta * alpha
            s_arr[k] = s
            d_arr[k] = d

        # cartesian
        xs = np.empty(n)
        ys = np.empty(n)
        for k in range(n):
            xs[k], ys[k] = self.to_cartesian(s_arr[k], d_arr[k])

        # vx
        v_base = np.array([self._vx_at(s_arr[k]) for k in range(n)])
        vx = v_base.copy()
        if v_cap is not None:
            vx = np.minimum(vx, float(v_cap))
        if use_curvature_cap:
            _, kappa_g = geom_psi_kappa(xs, ys)
            v_curv = np.sqrt(self.a_lat_max / np.maximum(np.abs(kappa_g), 1e-6))
            vx = np.minimum(vx, v_curv)
        if vx_scale != 1.0:
            vx = vx * float(vx_scale)
        if self._front_stop:
            vx = np.zeros_like(vx)

        # build Wpnt array
        out = self._empty_header()
        for k in range(n):
            psi, kp = self._psi_kappa_at(s_arr[k])
            w = Wpnt()
            w.id          = int(k)
            w.s_m         = float(s_arr[k])
            w.d_m         = float(d_arr[k])
            w.x_m         = float(xs[k])
            w.y_m         = float(ys[k])
            w.psi_rad     = float(psi)
            w.kappa_radpm = float(kp)
            w.vx_mps      = float(vx[k])
            w.ax_mps2     = 0.0
            w.d_right     = 0.0
            w.d_left      = 0.0
            out.wpnts.append(w)
        return out

    def _avoid_d_at(self, s, st):
        """Evaluate the committed avoidance cubic spline at s (wrap-safe)."""
        s_abs = st['ego_s_init'] + (s - st['ego_s_init']) % self.s_total
        if s_abs <= st['ego_s_init']:
            return st['ego_d_init']
        if s_abs >= st['s_d']:
            return 0.0
        return float(st['cs'](s_abs))

    def _build_passthrough(self):
        """Raceline (d_target = 0) published as-is (no ego blend).

        Ego-blend made the near-field reference follow the car's current
        lateral position, which zeroed out PP's CTE and let the car cut the
        inside of corners indefinitely. Publishing the pure raceline keeps CTE
        meaningful so PP's Kp_cte pulls the car back onto the line.
        """
        return self._make_local_wpnts(target_fn=lambda s: 0.0, use_blend=False)

    def _build_trailing(self):
        """Passthrough + trailing PD speed cap."""
        ego = (self.ex, self.ey, self.eyaw)
        v_cap = trailing(self.track, ego, self.ev)
        return self._make_local_wpnts(target_fn=lambda s: 0.0, v_cap=v_cap, use_blend=False)

    def _build_from_avoid_state(self, st):
        """Publish the committed avoidance spline as-is (no ego blend)."""
        return self._make_local_wpnts(
            target_fn=lambda s: self._avoid_d_at(s, st),
            use_curvature_cap=True,
            vx_scale=self.vx_scale_avoid,
            use_blend=False)

    def _build_spline_avoid_or_fallback(self):
        """Trailing only (avoidance disabled).
        track 유무 관계없이 항상 trailing() 경유 → base_speed 캡 적용.
        """
        return self._build_trailing(), 'trailing'

    # ================================================================== #
    # Avoidance state + feasibility
    # ================================================================== #
    def _make_avoidance_state(self, s_obs_rel, d_avoid, label, d_obs=0.0):
        """3-ctrl cubic spline, then push samples out of obstacle + clamp to
        walls + PCHIP refit.

        ctrl points: (ego_s_init, ego_d_init), (s_obs_rel, d_avoid),
                     (s_obs_rel + s_out, 0).
        s_d = last ctrl = commit termination check point.

        Post-processing (clamp_to_walls):
          1. dense-sample the raw cubic over [ego_s_init, s_d]
          2. push d laterally so |d - d_obs| >= sqrt(safety^2 - (s - s_obs)^2)
             on the side selected by sign(d_avoid)        (obstacle clearance)
          3. clamp each d into [-dr(s)+margin+buffer, dl(s)-margin-buffer]
             (wall clearance — final, so walls win over obstacle push if they
             collide; _evaluate_state then catches that as infeasible)
          4. refit with PCHIP (shape-preserving, no overshoot)
        """
        ego_s_init = float(self.ego_s)
        ego_d_init = float(self.ego_d)
        s_d = s_obs_rel + self.s_out

        # s_hold 구간 동안 현재 d 유지 → 가까운 구간 경로 변화 없음
        s_hold_end = ego_s_init + self.s_hold
        if s_obs_rel - s_hold_end > 0.5:
            s_ctrl = np.array([ego_s_init, s_hold_end, s_obs_rel, s_d], dtype=float)
            d_ctrl = np.array([ego_d_init, ego_d_init, float(d_avoid), 0.0], dtype=float)
        else:
            s_ctrl = np.array([ego_s_init, s_obs_rel, s_d], dtype=float)
            d_ctrl = np.array([ego_d_init, float(d_avoid), 0.0], dtype=float)
        if not np.all(np.diff(s_ctrl) > 1e-3):
            return None
        cs_raw = CubicSpline(s_ctrl, d_ctrl, bc_type='natural')

        if self.clamp_to_walls:
            n_clamp = 40
            s_seq = np.linspace(ego_s_init, s_d, n_clamp)
            d_seq = np.asarray(cs_raw(s_seq), dtype=float)
            inset  = self.margin + self.clamp_buffer
            safety = self.obs_radius + self.margin
            side = 1.0 if d_avoid > 0 else -1.0
            for i, s in enumerate(s_seq):
                d = d_seq[i]
                # 1) push outward of obstacle within its s-influence band
                ds_obs = s - s_obs_rel
                if abs(ds_obs) < safety:
                    lat_needed = math.sqrt(safety * safety - ds_obs * ds_obs)
                    target = d_obs + side * lat_needed
                    if side > 0:
                        d = max(d, target)
                    else:
                        d = min(d, target)
                # 2) wall clamp (final authority)
                dl =  self._dl_at(s) - inset
                dr = -self._dr_at(s) + inset
                if dl < dr:                       # corridor narrower than 2*inset
                    d = 0.5 * (dl + dr)
                else:
                    d = min(max(d, dr), dl)
                d_seq[i] = d
            cs = PchipInterpolator(s_seq, d_seq, extrapolate=False)
        else:
            cs = cs_raw

        return {
            'label':       label,
            'd_avoid':     float(d_avoid),
            's_d':         float(s_d),
            'ego_s_init':  ego_s_init,
            'ego_d_init':  ego_d_init,
            'cs':          cs,
        }

    def _sample_state_full(self, st, n=60):
        """Dense sampling over [ego_s_init, s_d] for feasibility & viz."""
        s_seq = np.linspace(st['ego_s_init'], st['s_d'], n)
        d_seq = np.array([self._avoid_d_at(s, st) for s in s_seq])
        return s_seq, d_seq

    def _evaluate_state(self, st, obs_x, obs_y):
        """Wall + obstacle feasibility + cost. Returns cost or None (infeasible)."""
        s_seq, d_seq = self._sample_state_full(st)
        # 1) track width (both walls + margin)
        for s, d in zip(s_seq, d_seq):
            dl = self._dl_at(s) - self.margin
            dr = -self._dr_at(s) + self.margin
            if d > dl or d < dr:
                return None
        # 2) min distance to obstacle (must clear inflated safety distance)
        xs = np.empty_like(s_seq)
        ys = np.empty_like(s_seq)
        for k, (s, d) in enumerate(zip(s_seq, d_seq)):
            xs[k], ys[k] = self.to_cartesian(s, d)
        obs_dist = float(np.min(np.hypot(xs - obs_x, ys - obs_y)))
        safety = self.obs_radius + self.margin
        if obs_dist < safety:
            return None
        # 3) cost: 1/clearance + mean |d|
        w_obs    = 5.0
        w_offset = 1.0
        return (w_obs / max(obs_dist - safety, 0.05)
                + w_offset * float(np.mean(np.abs(d_seq))))

    # ================================================================== #
    # Visualization
    # ================================================================== #
    def _publish_local_markers(self, wpnts, mode):
        color = (0.3, 0.8, 1.0)   # sky blue (always)
        ma = MarkerArray()
        line = Marker()
        line.header.frame_id = 'map'
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns = 'local_waypoints_line'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.08
        line.color.r, line.color.g, line.color.b, line.color.a = (*color, 1.0)
        # z above candidate (0.07) and global raceline so the local path
        # always draws on top in RViz.
        for w in wpnts.wpnts:
            p = Point()
            p.x, p.y, p.z = float(w.x_m), float(w.y_m), 0.20
            line.points.append(p)
        ma.markers.append(line)
        self.marker_pub.publish(ma)

    def _clear_candidates(self):
        """Publish DELETEALL to wipe leftover candidate markers."""
        ma = MarkerArray()
        clear = Marker(); clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        self.cand_pub.publish(ma)

    def _publish_candidates(self, results):
        """Visualize [{'state': st, 'cost': float|None}, ...]."""
        ma = MarkerArray()
        clear = Marker(); clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        feasible = [r for r in results if r['cost'] is not None]
        best_cost = min((r['cost'] for r in feasible), default=None)
        stamp = self.get_clock().now().to_msg()
        for i, r in enumerate(results):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'avoid_candidates'
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            if r['cost'] is None:
                # infeasible
                m.scale.x = 0.03
                m.color.r, m.color.g, m.color.b, m.color.a = 0.7, 0.0, 0.0, 0.4
            elif best_cost is not None and r['cost'] == best_cost:
                # best
                m.scale.x = 0.08
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 0.95
            else:
                # feasible but not best
                m.scale.x = 0.03
                m.color.r, m.color.g, m.color.b, m.color.a = 0.55, 0.55, 0.55, 0.6
            s_seq, d_seq = self._sample_state_full(r['state'])
            for s, d in zip(s_seq, d_seq):
                x, y = self.to_cartesian(s, d)
                p = Point(); p.x, p.y, p.z = float(x), float(y), 0.07
                m.points.append(p)
            # lifetime=0 -> persist in RViz until explicit DELETEALL.
            # We only clear when ego passes s_d (avoidance complete).
            m.lifetime.sec = 0
            m.lifetime.nanosec = 0
            ma.markers.append(m)
        self.cand_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = LocalPlanning()
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
