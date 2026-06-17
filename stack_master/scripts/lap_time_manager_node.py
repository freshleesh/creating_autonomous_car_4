#!/usr/bin/env python3
"""Lap time manager.

차량의 융합 localization pose(`/car_state/odom`)를 보며 start/finish 라인을 통과할
때마다 한 바퀴(lap)로 세고 lap time을 측정/발행한다.

라인 정의:
  start_x, start_y 의 한 점을 지나고, 주행 방향(start_yaw)에 수직인 직선.
  start_x 가 설정되지 않으면(기본) 첫 odom 메시지의 위치/방향을 시작점으로 캡처한다.

크로싱 판정:
  주행 방향 축으로의 부호거리 d 가 음(-)→양(+) 으로 바뀌고, 라인 가로 오프셋이
  half_width 안일 때 통과로 본다. 라인 근처 떨림으로 중복 카운트되는 것을 막기 위해,
  직전 통과 이후 차량이 라인 뒤쪽으로 arm_dist 이상 멀어졌을 때만(armed) 카운트하고
  min_lap_time 보다 빠른 통과는 무시한다.

타이밍은 odom 헤더 스탬프 기준이라 use_sim_time 에서도 정상 동작한다.

발행 토픽:
  ~/lap_time   (std_msgs/Float32)  방금 끝난 lap 의 소요 시간 [s]
  ~/best_lap   (std_msgs/Float32)  현재까지 최단 lap [s]
  ~/lap_count  (std_msgs/Int32)    완료한 lap 수
  ~/current_lap_path   (nav_msgs/Path)  현재 주행 중인 바퀴 경로
  ~/previous_lap_path  (nav_msgs/Path)  직전 완주 바퀴 경로 (latched)
  ~/markers    (visualization_msgs/MarkerArray)  RViz 용 라인 + 텍스트 + 경로
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, PoseStamped
from std_msgs.msg import Float32, Int32
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped


def yaw_from_quat(q) -> float:
    """Quaternion -> yaw (2D)."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class LapTimeManager(Node):
    def __init__(self):
        super().__init__('lap_time_manager')

        # --- topics / frames ---
        self.odom_topic = self.declare_parameter('odom_topic', '/car_state/odom').value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        # 컨트롤러 드라이브 명령(steer 읽기용).
        self.drive_topic = self.declare_parameter(
            'drive_topic', '/vesc/high_level/ackermann_cmd').value

        # --- start/finish line ---
        # start_x 가 유한하지 않으면 첫 odom 에서 자동 캡처한다.
        self.start_x = self.declare_parameter('start_x', float('nan')).value
        self.start_y = self.declare_parameter('start_y', float('nan')).value
        self.start_yaw = self.declare_parameter('start_yaw', float('nan')).value
        self.auto_start = self.declare_parameter('auto_start', True).value

        # --- 판정 파라미터 ---
        self.half_width = self.declare_parameter('half_width', 3.0).value      # 라인 가로 반폭 [m]
        self.arm_dist = self.declare_parameter('arm_dist', 1.0).value          # 재무장 거리 [m]
        self.min_lap_time = self.declare_parameter('min_lap_time', 2.0).value  # 최소 lap 시간 [s]

        # --- 동작 옵션 ---
        # time_from_start=True: 출발 시점(라인이 정해진 첫 odom)부터 타이머를 시작해
        #   첫 통과를 Lap 1 로 센다. auto_start 처럼 출발점=결승점일 때 유효하다.
        # False: 첫 통과를 기준선으로만 쓰고(부분 구간 버림) 두 번째 통과부터 Lap 1.
        #   출발선과 다른 지점에서 출발할 때 정확하다.
        self.time_from_start = self.declare_parameter('time_from_start', True).value
        # /initialpose(RViz 2D Pose Estimate) 수신 시 lap 카운트를 초기화한다.
        self.reset_on_initpose = self.declare_parameter('reset_on_initialpose', True).value
        # 현재 바퀴 주행 경로를 Path 로 발행. 점 간 최소 간격으로 다운샘플.
        self.path_min_dist = self.declare_parameter('path_min_dist', 0.03).value   # [m] 작을수록 촘촘
        self.path_line_width = self.declare_parameter('path_line_width', 0.15).value  # [m] LINE_STRIP 두께
        # 직전 바퀴 경로 발행 여부 + 마커 두께.
        self.publish_prev_path = self.declare_parameter('publish_prev_path', True).value
        self.prev_path_line_width = self.declare_parameter('prev_path_line_width', 0.1).value  # [m]

        self.have_line = False
        self._init_line_from_params()

        # 상태
        self.prev_d = None        # 직전 부호거리
        self.armed = False        # 통과 카운트 가능 여부
        self.last_cross_t = None  # 직전 통과 시각 [s]
        self.lap_count = 0
        self.best_lap = None
        self.last_lap = None  # 직전(이전) lap 시간 [s]
        self.cur_speed = 0.0  # 현재 속도 [m/s] (odom twist)
        self.cur_steer = 0.0  # 현재 조향각 [rad] (드라이브 명령)

        # 현재 바퀴 경로 (map 프레임). 새 lap 마다 초기화.
        self.path = Path()
        self.path.header.frame_id = self.map_frame
        self._last_path_xy = None
        # 직전 완주 바퀴 경로 (완주 시점에 현재 경로를 스냅샷). None 이면 아직 없음.
        self.prev_path = None

        self.line_marker_cached = None

        # I/O
        self.sub = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)
        self.create_subscription(
            AckermannDriveStamped, self.drive_topic, self.drive_cb, 10)
        if self.reset_on_initpose:
            self.create_subscription(
                PoseWithCovarianceStamped, '/initialpose', self.initpose_cb, 10)

        self.pub_lap = self.create_publisher(Float32, '~/lap_time', 10)
        self.pub_best = self.create_publisher(Float32, '~/best_lap', 10)
        self.pub_count = self.create_publisher(Int32, '~/lap_count', 10)
        self.pub_marker = self.create_publisher(MarkerArray, '~/markers', 1)
        self.pub_path = self.create_publisher(Path, '~/current_lap_path', 10)
        # 직전 바퀴 경로는 완주 때 한 번만 갱신되므로 latched(transient_local)로 발행해
        # RViz 가 늦게 구독해도 마지막 경로를 받도록 한다.
        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_prev_path = self.create_publisher(
            Path, '~/previous_lap_path', latched_qos)
        self.pub_speed = self.create_publisher(Float32, '~/speed', 10)  # [m/s]
        self.pub_steer = self.create_publisher(Float32, '~/steer', 10)  # [rad]

        # 마커는 주기적으로(2Hz) 갱신 발행
        self.create_timer(0.5, self.publish_markers)

        self.get_logger().info(
            f'lap_time_manager started: odom={self.odom_topic} '
            f'half_width={self.half_width} arm_dist={self.arm_dist} '
            f'min_lap_time={self.min_lap_time}')

    # ------------------------------------------------------------------ line
    def _init_line_from_params(self):
        if all(math.isfinite(v) for v in (self.start_x, self.start_y, self.start_yaw)):
            self._set_line(self.start_x, self.start_y, self.start_yaw)
            self.get_logger().info(
                f'start/finish line from params: '
                f'({self.start_x:.2f}, {self.start_y:.2f}, {self.start_yaw:.2f} rad)')

    def _set_line(self, x, y, yaw):
        self.start_x, self.start_y, self.start_yaw = x, y, yaw
        # n: 주행 방향(= 라인 법선), t: 라인을 따라가는 방향
        self.nx, self.ny = math.cos(yaw), math.sin(yaw)
        self.tx, self.ty = -math.sin(yaw), math.cos(yaw)
        self.have_line = True
        self.line_marker_cached = None  # 다음 publish 때 재생성

    # ----------------------------------------------------------------- timing
    @staticmethod
    def _stamp_sec(stamp) -> float:
        return Time.from_msg(stamp).nanoseconds * 1e-9

    # -------------------------------------------------------------- callbacks
    def drive_cb(self, msg: AckermannDriveStamped):
        self.cur_steer = msg.drive.steering_angle
        self.pub_steer.publish(Float32(data=float(self.cur_steer)))

    def odom_cb(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y

        self.cur_speed = msg.twist.twist.linear.x
        self.pub_speed.publish(Float32(data=float(self.cur_speed)))

        t = self._stamp_sec(msg.header.stamp)

        if not self.have_line:
            if not self.auto_start:
                return
            yaw = yaw_from_quat(msg.pose.pose.orientation)
            self._set_line(px, py, yaw)
            # 출발 시점부터 타이머 시작: 첫 통과가 곧 Lap 1 의 완주 시점이 된다.
            if self.time_from_start:
                self.last_cross_t = t
            self.get_logger().info(
                f'start/finish line auto-captured at '
                f'({px:.2f}, {py:.2f}, {yaw:.2f} rad)'
                + ('  — 타이머 시작' if self.time_from_start else ''))
            return

        # 파라미터로 라인을 지정한 경우: 처음 받은 odom 시각을 출발 기준으로 삼는다.
        if self.time_from_start and self.last_cross_t is None:
            self.last_cross_t = t
            self.get_logger().info('출발 시점부터 lap 타이머 시작')

        dx, dy = px - self.start_x, py - self.start_y
        d = dx * self.nx + dy * self.ny      # 주행 방향 부호거리
        lat = dx * self.tx + dy * self.ty    # 라인 가로 오프셋

        # 라인 뒤쪽으로 충분히 멀어지면 다음 통과를 받을 수 있게 재무장
        if d < -self.arm_dist:
            self.armed = True

        if (self.prev_d is not None and self.prev_d < 0.0 and d >= 0.0
                and abs(lat) <= self.half_width and self.armed):
            self._register_crossing(t)  # 새 lap 이면 경로 초기화됨

        self.prev_d = d
        # 현재 바퀴 경로 누적 + 실시간 발행 (크로싱 후라 새 lap 첫 점부터 담김)
        self._update_path(px, py, msg)

    def _register_crossing(self, t: float):
        self.armed = False

        if self.last_cross_t is None:
            # time_from_start=False 일 때만 도달: 첫 통과를 기준선으로만 쓰고
            # (부분 구간이라 버림) 다음 통과부터 Lap 을 센다.
            self.last_cross_t = t
            self._reset_path()  # 여기서부터가 Lap 1 의 시작점
            self.get_logger().info('start/finish 첫 통과 — lap 타이머 시작')
            return

        lap = t - self.last_cross_t
        if lap < self.min_lap_time:
            # 너무 빠른 통과(떨림/역주행 노이즈)는 무시, 기준 시각도 유지
            self.get_logger().warn(
                f'통과 무시: lap {lap:.2f}s < min_lap_time {self.min_lap_time:.2f}s')
            self.armed = False
            return

        self.last_cross_t = t
        self.lap_count += 1
        self.last_lap = lap
        if self.best_lap is None or lap < self.best_lap:
            self.best_lap = lap
        delta = '' if self.best_lap is None else f'  (best {self.best_lap:.3f}s)'
        self.get_logger().info(f'Lap {self.lap_count}: {lap:.3f}s{delta}')

        self.pub_lap.publish(Float32(data=float(lap)))
        self.pub_best.publish(Float32(data=float(self.best_lap)))
        self.pub_count.publish(Int32(data=int(self.lap_count)))

        # 방금 끝난 바퀴 경로를 직전 경로로 스냅샷 후 발행, 그다음 현재 경로 초기화.
        self._snapshot_prev_path()
        self._reset_path()  # 다음 바퀴 경로 새로 시작

    def initpose_cb(self, _msg: PoseWithCovarianceStamped):
        self.prev_d = None
        self.armed = False
        self.last_cross_t = None
        self.lap_count = 0
        self.best_lap = None
        self.last_lap = None
        self._reset_path()
        self.prev_path = None
        if self.publish_prev_path:
            self.pub_prev_path.publish(Path(header=self.path.header))  # 빈 경로로 클리어
        self.get_logger().info('/initialpose 수신 — lap 카운트 초기화')

    # ------------------------------------------------------------------ path
    def _reset_path(self):
        self.path.poses = []
        self._last_path_xy = None

    def _snapshot_prev_path(self):
        # 방금 끝난 바퀴 경로를 복사해 직전 경로로 보관하고 latched 토픽으로 발행.
        if not self.publish_prev_path or not self.path.poses:
            return
        prev = Path()
        prev.header.frame_id = self.map_frame
        prev.header.stamp = self.path.header.stamp
        prev.poses = list(self.path.poses)  # 새 lap 의 reset 이 영향 주지 않게 얕은 복사
        self.prev_path = prev
        self.pub_prev_path.publish(prev)

    def _update_path(self, px, py, msg: Odometry):
        # 직전 점에서 path_min_dist 이상 움직였을 때만 점 추가(다운샘플).
        if (self._last_path_xy is None or
                math.hypot(px - self._last_path_xy[0],
                           py - self._last_path_xy[1]) >= self.path_min_dist):
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            ps.header.stamp = msg.header.stamp
            ps.pose.position.x = px
            ps.pose.position.y = py
            ps.pose.orientation = msg.pose.pose.orientation
            self.path.poses.append(ps)
            self._last_path_xy = (px, py)
        self.path.header.stamp = msg.header.stamp
        self.pub_path.publish(self.path)
        self._publish_path_marker(msg.header.stamp)

    def _publish_path_marker(self, stamp):
        # 두께를 코드로 보장하기 위해 LINE_STRIP 마커로도 발행 (Path 는 RViz 설정에
        # 두께가 좌우됨). 색은 start/finish 라인(노랑)과 구분되게 초록.
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = stamp
        m.ns = 'lap_path'
        m.id = 2
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = self.path_line_width
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.2, 1.0
        m.points = [p.pose.position for p in self.path.poses]
        self.pub_marker.publish(MarkerArray(markers=[m]))

    def _prev_path_marker(self, stamp):
        # 직전 바퀴 경로를 파랑 LINE_STRIP 으로. (현재 경로=초록, 라인=노랑과 구분)
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = stamp
        m.ns = 'prev_lap_path'
        m.id = 3
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = self.prev_path_line_width
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.4, 1.0, 1.0
        m.points = [p.pose.position for p in self.prev_path.poses]
        return m

    # ----------------------------------------------------------------- markers
    def publish_markers(self):
        if not self.have_line:
            return
        arr = MarkerArray()

        # 직전 바퀴 경로 마커는 정적이라 2Hz 타이머에서 함께 재발행한다.
        if self.publish_prev_path and self.prev_path is not None:
            arr.markers.append(
                self._prev_path_marker(self.get_clock().now().to_msg()))

        line = Marker()
        line.header.frame_id = self.map_frame
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns = 'lap_line'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.1
        line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 1.0, 0.0, 1.0
        p1 = Point(x=self.start_x + self.tx * self.half_width,
                   y=self.start_y + self.ty * self.half_width, z=0.0)
        p2 = Point(x=self.start_x - self.tx * self.half_width,
                   y=self.start_y - self.ty * self.half_width, z=0.0)
        line.points = [p1, p2]
        arr.markers.append(line)

        text = Marker()
        text.header = line.header
        text.ns = 'lap_text'
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = self.start_x
        text.pose.position.y = self.start_y
        text.pose.position.z = 0.5
        text.scale.z = 0.4
        text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 1.0, 1.0, 1.0
        last = '-' if self.last_lap is None else f'{self.last_lap:.2f}s'
        best = '-' if self.best_lap is None else f'{self.best_lap:.2f}s'
        text.text = (
            f'laps: {self.lap_count}  last: {last}  best: {best}\n'
            f'v: {self.cur_speed:.2f} m/s  steer: {math.degrees(self.cur_steer):+.1f} deg')
        arr.markers.append(text)

        self.pub_marker.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = LapTimeManager()
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
