#!/usr/bin/env python3

#로컬플래닝_정적장애물v2

"""
local_planning.py 

perception + Frenet + planner

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
import os
import yaml as _yaml

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import binary_dilation

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
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
    r_max = 7.0     # [m] ignore returns farther than this

    n = ranges.shape[0]
    angles = angle_min + np.arange(n) * angle_inc
    valid = np.isfinite(ranges) & (ranges >= r_min) & (ranges <= r_max)
    x = np.where(valid, ranges * np.cos(angles), np.nan)
    y = np.where(valid, ranges * np.sin(angles), np.nan)
    return x, y
 


# ===========================================================================
#  PERCEPTION 
# ===========================================================================


def cluster(x: np.ndarray, y: np.ndarray, angle_inc: float):
    
    # --- tunable parameters ---
    lambda_rad = math.radians(30.0)
    sigma      = 0.5  # 클수록 더 클러스터링 잘되게
    min_points = 10      # 클러스터링 충족하는 포인트 수 (원거리 감지 대응)


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
    """정적 장애물용 단순 클러스터 중심 검출.
    L-shape/eigenvalue 조건 없음 — 크기 필터만 적용."""
    max_obs_size = 1.25   # [m] 이보다 큰 클러스터는 벽으로 간주
    obstacles    = []

    for cl in clusters:
        pts = np.array(cl, dtype=float)
        if pts.shape[0] < 3:
            continue
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        # 클러스터 최대 스팬 = 외접원 지름
        span = float(np.max(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy))) * 2.0
        if span > max_obs_size:
            continue
        obstacles.append((cx, cy, 0.50, 0.35))

    return obstacles









def tracking(obstacles, track, dt: float, ego, max_misses: int = 8):
    # --- tunable parameters ---
    Q_scale         = 0.5     # 칼만필터 노이즈 공분산
    R_scale         = 0.5     # 측정 노이즈 가중치
    assoc_threshold = 1.0     # 동일한 물체로 인식하는 거리 기준 (코너 예측 오차 대응)
    lidar_to_base_x = 0.27    # 라이다 기준 좌표 거리 (scan_cb 필터와 동일)

    ex, ey, eyaw = ego

    # 트랙 있으면 트랙 위치 기준, 없으면 ego 기준으로 가장 가까운 측정값 선택
    ref_x = float(track[0][0]) if track is not None else ex
    ref_y = float(track[0][1]) if track is not None else ey

    meas = None
    best_dist = float('inf')
    for (cx_l, cy_l, w, h) in obstacles:
        if not (-0.5 < cx_l < 15.0):
            continue
        bx = cx_l + lidar_to_base_x
        by = cy_l
        gx = ex + bx * math.cos(eyaw) - by * math.sin(eyaw)
        gy = ey + bx * math.sin(eyaw) + by * math.cos(eyaw)

        dist = math.hypot(gx - ref_x, gy - ref_y)
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
    base_speed   = 8.0   # [m/s] free-running race speed
    desired_gap  = 1.0   # [m] PD 목표 거리
    detect_range = 4.0   # [m] 이 거리부터 PD 감속 시작
    stop_gap     = 2.0   # [m] 이 거리 이내 → 완전 정지
    kp           = 3.0   # P gain: gap 오차 1m당 속도 보정
    kd           = 4.0   # D gain: closing speed 1m/s당 보정
    max_speed    = 6.0   # [m/s] absolute speed cap





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

    # 5. PD speed
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
        self.d_safe        = float(gp('d_safe',       0.6))     # [m] lateral offset
        self.s_in          = float(gp('s_in',         1.5))     # [m] min approach gap
        self.s_out         = float(gp('s_out',        4.0))     # [m] peak -> raceline
        self.trigger_range = float(gp('trigger_range', 1.5))    # [m] forward trigger range
        self.margin        = float(gp('margin',       0.1))     # [m] wall/obstacle margin
        self.obs_radius    = float(gp('obs_radius',   0.4))     # [m] obstacle inflate
        self.track_half_w  = float(gp('track_half_w', 0.8))     # [m] fallback half-width
        self.a_lat_max     = float(gp('a_lat_max',    6.0))     # [m/s^2] lat accel cap
        self.vx_scale_avoid = float(gp('vx_scale_avoid', 0.7))  # avoidance vx multiplier
        self.vx_max        = float(gp('vx_max',       99.0))    # [m/s]   전체 속도 상한

        # ---- wall clamping (per-sample clamp + PCHIP refit) -----------------
        self.clamp_to_walls = bool(gp('clamp_to_walls', True))
        self.clamp_buffer   = float(gp('clamp_buffer',  0.05))     # [m]

        # ---- avoidance shape ------------------------------------------------
        self.s_hold = float(gp('s_hold', 1.0))  # [m] 가까운 구간 경로 유지 거리



        # ---- PNG wall mask --------------------------------------------------
        _map_name = str(gp('map_name', ''))
        _default_png = ''
        _default_res, _default_ox, _default_oy = 0.050, 0.0, 0.0
        if _map_name:
            try:
                _pkg = get_package_share_directory('stack_master')
                _map_yaml_path = os.path.join(_pkg, 'maps', _map_name, f'{_map_name}.yaml')
                with open(_map_yaml_path) as _f:
                    _mi = _yaml.safe_load(_f)
                _default_res = float(_mi.get('resolution', 0.050))
                _default_ox  = float(_mi['origin'][0])
                _default_oy  = float(_mi['origin'][1])
                _default_png = os.path.join(_pkg, 'maps', _map_name, f'{_map_name}.png')
            except Exception as _e:
                self.get_logger().warn(f'PNG wall mask load failed: {_e}')
        self._wall_png  = str(gp('map_png', _default_png))
        self._wall_res  = float(gp('map_res', _default_res))
        self._wall_ox   = float(gp('map_ox',  _default_ox))
        self._wall_oy   = float(gp('map_oy',  _default_oy))
        self._wall_mask       = None   # 10cm — 스캔 포인트 필터
        self._wall_mask_large = None   # 20cm — track 벽 반사 소멸
        self._wall_h    = 0
        self._wall_w    = 0
        self._load_wall_mask()



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
        self.ex = self.ey = self.eyaw = 0.0
        self.ev = 0.0
        self.have_pose = False
        self._yaw_rate  = 0.0   # [rad/s] odom angular.z 직접 사용
        self.ego_s = 0.0
        self.ego_d = 0.0

        # ---- avoidance commit (hysteresis) ----------------------------------
        self._avoid_state = None

        # ---- ROS interfaces -------------------------------------------------
        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(WpntArray,  self.global_topic, self._global_wp_cb, latched)
        self.create_subscription(Odometry,   self.odom_topic,   self._odom_cb, 10)
        self.create_subscription(LaserScan,  self.scan_topic,   self._scan_cb, 10)

        self.local_pub  = self.create_publisher(WpntArray,   self.local_topic, latched)
        self.viz_pub    = self.create_publisher(MarkerArray, '/local_planning/viz', 10)
        self.cand_pub   = self.create_publisher(MarkerArray, '/local_planning/candidates', 5)
        self.det_pub    = self.create_publisher(MarkerArray, '/local_planning/detections', 5)

        # 트리거 콘 1 (직선용): 좁고 멀리
        self._BOX_LEN        = 5.0              # [m] 콘 반경
        self._CONE_HALF_DEG  = 17.0             # [deg] 콘 반각 (총 34°)
        # 트리거 콘 2 (코너용): 넓고 가까이 — 두 부채꼴 OR 조건
        self._BOX_LEN_WIDE   = 2.5              # [m] 넓은 콘 반경
        self._CONE_HALF_DEG_WIDE = 40.0         # [deg] 넓은 콘 반각 (총 80°)
        self._MIN_HITS       = 5                # avoidance 발동 최소 연속 감지 횟수

        self.get_logger().info(
            f'local_planning up | mode={self.mode} | horizon={self.local_horizon} m | '
            f'global={self.global_topic} -> local={self.local_topic}')

    # ================================================================== #
    # PNG wall mask
    # ================================================================== #
    def _load_wall_mask(self):
        """PNG 맵에서 검은색 벽 픽셀을 읽어 두 가지 마진의 이진 마스크를 생성.

        _wall_mask       (10cm) : 스캔 포인트 필터
        _wall_mask_large (20cm) : track 벽 반사 소멸
        """
        if not self._wall_png or not os.path.isfile(self._wall_png):
            self.get_logger().warn(
                f'wall mask: PNG 경로 없음/누락 (map_name 미지정?) ({self._wall_png!r}) — 벽 마스크 비활성화')
            return
        arr = None
        try:
            from PIL import Image
            img = Image.open(self._wall_png).convert('L')
            arr = np.array(img, dtype=np.uint8)
        except ImportError:
            try:
                import cv2
                arr = cv2.imread(self._wall_png, cv2.IMREAD_GRAYSCALE)
            except Exception as _e:
                self.get_logger().warn(f'wall mask: cv2 로드 실패 ({_e})')
        except Exception as _e:
            self.get_logger().warn(f'wall mask: PNG 로드 실패 ({_e})')
        if arr is None:
            self.get_logger().warn(f'wall mask: PNG 로드 실패 ({self._wall_png})')
            return
        wall = arr < 90
        n_sm = max(1, int(math.ceil(0.20 / self._wall_res)))   # 20cm — 스캔/클러스터 필터
        n_lg = max(1, int(math.ceil(0.30 / self._wall_res)))   # 30cm — track kill
        self._wall_mask       = binary_dilation(wall, structure=np.ones((2*n_sm+1, 2*n_sm+1), dtype=bool))
        self._wall_mask_large = binary_dilation(wall, structure=np.ones((2*n_lg+1, 2*n_lg+1), dtype=bool))
        self._wall_h, self._wall_w = arr.shape
        self.get_logger().info(
            f'wall mask ready: {self._wall_h}×{self._wall_w} px (20cm / 30cm)')

    def _is_on_wall_png(self, gx: float, gy: float) -> bool:
        """글로벌 좌표 (gx, gy)가 PNG 벽 마스크 위이면 True."""
        if self._wall_mask is None:
            return False
        col = int((gx - self._wall_ox) / self._wall_res)
        row = int(self._wall_h - 1 - (gy - self._wall_oy) / self._wall_res)
        if 0 <= row < self._wall_h and 0 <= col < self._wall_w:
            return bool(self._wall_mask[row, col])
        return False

    # ================================================================== #
    # ROS callbacks
    # ================================================================== #
    def _remove_wall_scan_points(self, x, y):
        """PNG 벽 마스크(10cm)로 벽 스캔 포인트를 NaN 처리."""
        if self._wall_mask is None:
            return x, y
        lidar_to_base_x = 0.27
        cyaw = math.cos(self.eyaw)
        syaw = math.sin(self.eyaw)
        valid = np.isfinite(x) & np.isfinite(y)
        bx = np.where(valid, x + lidar_to_base_x, 0.0)
        by = np.where(valid, y, 0.0)
        gx = self.ex + cyaw * bx - syaw * by
        gy = self.ey + syaw * bx + cyaw * by
        cols = ((gx - self._wall_ox) / self._wall_res).astype(int)
        rows = (self._wall_h - 1 - (gy - self._wall_oy) / self._wall_res).astype(int)
        in_bounds = valid & (cols >= 0) & (cols < self._wall_w) & (rows >= 0) & (rows < self._wall_h)
        rc = np.clip(rows, 0, self._wall_h - 1)
        cc = np.clip(cols, 0, self._wall_w - 1)
        on_wall = in_bounds & self._wall_mask[rc, cc]
        x = x.copy(); y = y.copy()
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
        self.ev = msg.twist.twist.linear.x
        self._yaw_rate = msg.twist.twist.angular.z  # 바퀴 방향 추정용 yaw rate
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
        x, y = self._remove_wall_scan_points(x, y)
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
            # PNG 벽 마스크 필터: 벽 5cm 이내 → 정적 장애물로 제거
            if self._is_on_wall_png(gx, gy):
                continue

            # Frenet 필터: 트랙 경계 안쪽 버퍼 이외는 제거 (벽 반사 제거)
            if self._sx is not None:
                s_obs, d_obs = self.to_frenet(gx, gy)
                wall_buf = 0.20
                if d_obs > self._dl_at(s_obs) - wall_buf or d_obs < -self._dr_at(s_obs) + wall_buf:
                    continue

            dynamic_obstacles.append((cx_l, cy_l, w, h))
        obstacles = dynamic_obstacles

        ego = (self.ex, self.ey, self.eyaw)
        self.track = tracking(obstacles, self.track, dt, ego)

        # track 상태 벽 체크: 20cm 이내 + 실제 측정 없는(misses>5) 경우만 소멸
        if self.track is not None:
            state, P, misses, hits = self.track
            tx, ty = float(state[0]), float(state[1])
            col = int((tx - self._wall_ox) / self._wall_res)
            row = int(self._wall_h - 1 - (ty - self._wall_oy) / self._wall_res)
            in_bounds = 0 <= row < self._wall_h and 0 <= col < self._wall_w
            if in_bounds and self._wall_mask_large is not None and self._wall_mask_large[row, col]:
                if misses > 5 and self._avoid_state is None:  # 측정 없는 상태에서 벽 안쪽 → 벽 반사
                    self.track = None

        # 1.5) Detection 마커 (별도 토픽 유지)
        self._publish_detection_markers(obstacles)

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

        # 통합 마커: trigger_zone + tracking + local_waypoints 한 번에 발행
        self._publish_viz(out, used_mode)

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

    def _tracking_markers(self) -> list:
        markers = []
        if self.track is None or not self._is_in_trigger_box():
            return markers
        state, P, misses, hits = self.track
        gx, gy, vx, vy = state
        stamp = self.get_clock().now().to_msg()
        m = Marker()
        m.header.frame_id = 'map'; m.header.stamp = stamp
        m.ns = 'tracking_pos'; m.id = 100
        m.type = Marker.CYLINDER; m.action = Marker.ADD
        m.pose.position.x = float(gx); m.pose.position.y = float(gy); m.pose.position.z = 0.2
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.5
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 1.0, 0.8
        markers.append(m)
        speed = math.hypot(vx, vy)
        if speed > 0.02:
            arrow_len = max(speed * 3.0, 0.5)
            nx = vx / speed; ny = vy / speed
            arr = Marker()
            arr.header.frame_id = 'map'; arr.header.stamp = stamp
            arr.ns = 'tracking_vel'; arr.id = 101
            arr.type = Marker.ARROW; arr.action = Marker.ADD
            p1 = Point(); p1.x, p1.y, p1.z = float(gx), float(gy), 0.3
            p2 = Point(); p2.x = float(gx + nx * arrow_len); p2.y = float(gy + ny * arrow_len); p2.z = 0.3
            arr.points = [p1, p2]
            arr.scale.x = 0.08; arr.scale.y = 0.20; arr.scale.z = 0.0
            arr.color.r, arr.color.g, arr.color.b, arr.color.a = 1.0, 0.4, 0.7, 1.0
            markers.append(arr)
        return markers

    def _cone_half_deg(self) -> float:
        return self._CONE_HALF_DEG

    def _cone_yaw(self) -> float:
        """콘 방향: 레이스라인 2m 앞 지점의 heading 사용.
        코너 진입 전에 미리 돌아가서 화끈하게 예측."""
        if self._sx is not None and self.s_total > 0.0:
            psi, _ = self._psi_kappa_at(self.ego_s + 0.8)
            return psi
        return self.eyaw

    def _in_cone(self, fwd: float, lat: float, half_deg: float, radius: float) -> bool:
        """단일 부채꼴 + 후방 박스 판정 헬퍼."""
        near_half = 0.5
        if fwd < 0.0 or fwd > radius:
            return False
        lat_limit = max(fwd * math.tan(math.radians(half_deg)), near_half)
        return abs(lat) <= lat_limit

    def _is_in_trigger_box(self) -> bool:
        """두 부채꼴(직선용·코너용) 중 하나에 들어오면 True.
        콘 방향은 레이스라인 0.8m 앞 heading 기준."""
        if self.track is None:
            return False
        cone_yaw = self._cone_yaw()
        ox, oy = float(self.track[0][0]), float(self.track[0][1])
        cyaw = math.cos(cone_yaw)
        syaw = math.sin(cone_yaw)
        fwd = cyaw * (ox - self.ex) + syaw * (oy - self.ey)
        lat = -syaw * (ox - self.ex) + cyaw * (oy - self.ey)
        return (self._in_cone(fwd, lat, self._CONE_HALF_DEG,      self._BOX_LEN) or
                self._in_cone(fwd, lat, self._CONE_HALF_DEG_WIDE, self._BOX_LEN_WIDE))

    def _make_cone_marker(self, mid: int, half_deg: float, radius: float,
                          cone_yaw: float, color: tuple) -> Marker:
        """단일 부채꼴 + 후방 박스 TRIANGLE_LIST 마커 생성."""
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'trigger_zone'; m.id = mid
        m.type = Marker.TRIANGLE_LIST; m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = color
        N = 24
        half_rad = math.radians(half_deg)
        near_half = 0.5
        sc = math.cos(cone_yaw); ss = math.sin(cone_yaw)
        for i in range(N):
            t1 = i / N; t2 = (i + 1) / N
            pn1 = Point(); pn1.x = self.ex + (near_half - 2.0*near_half*t1)*ss; pn1.y = self.ey - (near_half - 2.0*near_half*t1)*sc; pn1.z = 0.05
            pn2 = Point(); pn2.x = self.ex + (near_half - 2.0*near_half*t2)*ss; pn2.y = self.ey - (near_half - 2.0*near_half*t2)*sc; pn2.z = 0.05
            a1 = cone_yaw - half_rad + 2.0*half_rad*t1; a2 = cone_yaw - half_rad + 2.0*half_rad*t2
            pf1 = Point(); pf1.x = self.ex + radius*math.cos(a1); pf1.y = self.ey + radius*math.sin(a1); pf1.z = 0.05
            pf2 = Point(); pf2.x = self.ex + radius*math.cos(a2); pf2.y = self.ey + radius*math.sin(a2); pf2.z = 0.05
            m.points += [pn1, pf1, pf2, pn1, pf2, pn2]
        return m

    def _trigger_zone_markers(self) -> list:
        """두 부채꼴 트리거 존 마커 반환. avoidance → 빨간색, free → 연두색."""
        cone_yaw = self._cone_yaw()
        if self._avoid_state is not None:
            c_far  = (1.0, 0.0, 0.0, 0.40)
            c_near = (1.0, 0.3, 0.0, 0.30)
        else:
            c_far  = (0.2, 1.0, 0.2, 0.25)
            c_near = (0.2, 1.0, 0.2, 0.18)
        m_far  = self._make_cone_marker(200, self._CONE_HALF_DEG,      self._BOX_LEN,       cone_yaw, c_far)
        m_near = self._make_cone_marker(201, self._CONE_HALF_DEG_WIDE, self._BOX_LEN_WIDE,  cone_yaw, c_near)
        return [m_far, m_near]

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
        vx = np.minimum(vx, self.vx_max)   # 전체 속도 상한
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
        """4m 이내 장애물 감지 시 좌/우 스플라인 회피, 불가능하면 trailing 폴백."""
        ego = (self.ex, self.ey, self.eyaw)

        # 1. 커밋된 회피 경로 진행 중
        if self._avoid_state is not None:
            st = self._avoid_state
            delta_ego = (self.ego_s - st['ego_s_init']) % self.s_total
            delta_sd  = st['s_d'] - st['ego_s_init']
            if delta_ego > delta_sd:
                # ego가 s_d를 지남 → 회피 완료
                self._avoid_state = None
                self._clear_candidates()
            else:
                return self._build_from_avoid_state(st), 'spline_avoid'

        # 2. track 없으면 free처럼 달리기
        if self.track is None:
            return self._build_passthrough(), 'free'

        # hits 부족 → 아직 충분히 감지되지 않은 물체 (유령 감지 방지)
        _, _, _, hits = self.track
        if hits < self._MIN_HITS:
            return self._build_passthrough(), 'free'

        # 3. 전방 거리 계산
        ox, oy = float(self.track[0][0]), float(self.track[0][1])
        cyaw = math.cos(self.eyaw)
        syaw = math.sin(self.eyaw)
        fwd_dist = cyaw * (ox - self.ex) + syaw * (oy - self.ey)
        lat_dist = -syaw * (ox - self.ex) + cyaw * (oy - self.ey)

        # 콘 트리거 밖이면 → free
        if not self._is_in_trigger_box():
            return self._build_passthrough(), 'free'

        # 4. Frenet 장애물 위치
        s_obs, d_obs = self.to_frenet(ox, oy)
        s_obs_rel = self.ego_s + (s_obs - self.ego_s) % self.s_total
        approach = s_obs_rel - self.ego_s

        # 너무 가까워서 경로 생성 불가 → trailing(속도 제어만)
        if approach < self.s_in:
            return self._build_trailing(), 'trailing'

        # 5. 좌/우 후보 생성 및 평가
        candidates = []
        for d_avoid, label in [(d_obs + self.d_safe, 'left'),
                                (d_obs - self.d_safe, 'right')]:
            st = self._make_avoidance_state(s_obs_rel, d_avoid, label, d_obs)
            cost = self._evaluate_state(st, ox, oy) if st is not None else None
            candidates.append({'state': st, 'cost': cost})
        self._publish_candidates(candidates)

        feasible = [c for c in candidates if c['cost'] is not None]
        if not feasible:
            return self._build_trailing(), 'trailing'

        # 6. 최적 후보 커밋
        best = min(feasible, key=lambda c: c['cost'])
        self._avoid_state = best['state']
        return self._build_from_avoid_state(self._avoid_state), 'spline_avoid'

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
    def _local_wp_markers(self, wpnts, mode) -> list:
        if mode == 'spline_avoid':
            color, width = (1.0, 0.0, 0.0), 0.12
        elif mode == 'trailing':
            color, width = (1.0, 1.0, 0.0), 0.08
        else:
            color, width = (0.2, 1.0, 0.2), 0.08
        line = Marker()
        line.header.frame_id = 'map'
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns = 'local_waypoints_line'; line.id = 300
        line.type = Marker.LINE_STRIP; line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = width
        line.color.r, line.color.g, line.color.b, line.color.a = (*color, 1.0)
        for w in wpnts.wpnts:
            p = Point(); p.x, p.y, p.z = float(w.x_m), float(w.y_m), 0.20
            line.points.append(p)
        return [line]

    def _publish_viz(self, wpnts, mode):
        """trigger_zone + tracking + local_waypoints 를 하나의 MarkerArray로 발행."""
        ma = MarkerArray()
        clear = Marker(); clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        ma.markers += self._trigger_zone_markers()
        ma.markers += self._tracking_markers()
        ma.markers += self._local_wp_markers(wpnts, mode)
        self.viz_pub.publish(ma)

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
