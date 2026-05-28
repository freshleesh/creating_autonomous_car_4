import math
import heapq
import time
from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped

from controller.estop import EStop


class GapFollowNode(Node):

    # ── 파라미터 ──────────────────────────────────────────────
    GRID_RESOLUTION = 0.1        # m/cell
    GRID_WIDTH = 100              # cells, 전방 x
    GRID_HEIGHT = 100             # cells, 측면 y
    GRID_REAR = 5                # cells, 후방 x
    MAX_SCAN_RANGE = 10.0         # m
    SCAN_FOV = math.pi * 1 / 2        # rad (정면 ±90도)
    RAY_COUNT = 360               # 레이캐스팅 빔 수 (정면 180도)
    HARD_INFLATION = 0.2          # m, scan point 주변 100으로 채우는 반경
    SOFT_INFLATION = 0.6          # m, 정규분포 cost 확장 반경
    LASER_OFFSET_X = 0.0          # m, laser→base_link x 오프셋
    WHEELBASE = 0.33              # m
    MAX_STEER = 0.4               # rad
    SPEED = 6.0                   # m/s
    SPEED_MIN = 2.0               # m/s (최대 조향 시)
    ASTAR_MAX_ITER = 15000        # A* 최대 탐색 노드 수
    INFLATION_COST_SCALE = 3.0    # inflate 셀 cost 배율 (1~99 → 1x ~ Nx)
    UNKNOWN_COST = 3.0            # unknown(-1) 셀 통과 시 추가 cost 배율
    PP_LOOKAHEAD = 1.5            # m
    MARKER_SPHERE_SIZE = 0.3      # m
    MARKER_LINE_WIDTH = 0.05      # m
    SPLINE_SKIP = 5               # A* 셀 몇 개마다 spline 제어점 선택
    SPLINE_DENSITY = 8            # spline 세그먼트당 출력 샘플 수

    def __init__(self):
        super().__init__('gap_follow')

        self.estop = EStop(self)
        self.total_w = self.GRID_WIDTH + self.GRID_REAR

        # Hard inflation kernel (binary circle)
        hard_cells = int(math.ceil(self.HARD_INFLATION / self.GRID_RESOLUTION))
        hy, hx = np.mgrid[-hard_cells:hard_cells+1, -hard_cells:hard_cells+1]
        self.hard_kernel = ((hx**2 + hy**2) <= hard_cells**2).astype(np.uint8)
        self.hard_cells = hard_cells

        # Soft inflation kernel (gaussian, 0~99)
        sigma_cells = self.SOFT_INFLATION / self.GRID_RESOLUTION / 2.0
        soft_cells = int(math.ceil(3.0 * sigma_cells))
        sy, sx = np.mgrid[-soft_cells:soft_cells+1, -soft_cells:soft_cells+1]
        gauss = np.exp(-(sx**2 + sy**2) / (2.0 * sigma_cells**2))
        self.soft_kernel = (gauss * 99).astype(np.int8)
        self.soft_cells = soft_cells

        # Raycasting 각도 배열 (정면 180도)
        self.ray_angles = np.linspace(-self.SCAN_FOV, self.SCAN_FOV, self.RAY_COUNT)
        self.ray_max_steps = int(self.MAX_SCAN_RANGE / self.GRID_RESOLUTION)

        self.odom = None

        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/vesc/odom', self._odom_cb, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/grid_map', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/pp_markers', 10)

        self.get_logger().info('GapFollowNode ready')

    def _scan_cb(self, msg):
        self.scan = msg
        if self.odom is None:
            return

        steer, speed = self._compute()
        drive = AckermannDriveStamped()
        drive.header.stamp = msg.header.stamp
        drive.header.frame_id = 'base_link'
        drive.drive.steering_angle = steer
        drive.drive.speed = speed
        self.drive_pub.publish(drive)

    def _odom_cb(self, msg):
        self.odom = msg

    def _compute(self):
        start = time.time()
        scan_stamp = self.scan.header.stamp
        now_stamp = self.get_clock().now().to_msg()

        grid = self._build_grid()
        self._publish_grid(grid, scan_stamp)

        robot_cell = self._world_to_grid(-self.LASER_OFFSET_X, 0.0)
        sr, sc = robot_cell

        start_cell = self._nearest_free_start(grid, sr, sc)
        goal_cell  = self._farthest_free(grid, sr, sc)

        if goal_cell is None:
            self.get_logger().info('A* skip: no free goal cell (all cells inflated or unknown)')
            return 0.0, self.SPEED

        goal_x, _ = self._grid_to_world(goal_cell[0], goal_cell[1])
        if goal_x > 10000.0:
            speed_override = 10.0
        else:
            speed_override = None

        path_cells = self._astar(grid, start_cell, goal_cell)
        if not path_cells or len(path_cells) < 2:
            self.get_logger().info(
                f'A* failed: start={start_cell} goal={goal_cell} '
                f'goal_val={int(grid[goal_cell[0], goal_cell[1]])}'
            )
            return 0.0, self.SPEED

        path_world = [self._grid_to_world(r, c) for r, c in path_cells]
        path_smooth = self._spline_path(path_world)

        steer, lookahead_pt = self._pure_pursuit(path_smooth)
        steer = float(np.clip(steer, -self.MAX_STEER, self.MAX_STEER))
        self._publish_markers(path_smooth, lookahead_pt, now_stamp)

        # 조향각에 비례해서 감속 (직선=SPEED, 최대조향=SPEED_MIN)
        if speed_override is not None:
            speed = speed_override
        else:
            ratio = abs(steer) / self.MAX_STEER
            speed = self.SPEED - ratio * (self.SPEED - self.SPEED_MIN)
        end = time.time()
        process = end - start
        self.get_logger().info(
            f'time: {process}')
        return steer, speed

    # ── Occupancy Grid (4-step pipeline) ──────────────────────

    def _build_grid(self):
        H, W = self.GRID_HEIGHT, self.total_w
        cy = H // 2
        res = self.GRID_RESOLUTION

        # Step 1: unknown grid
        grid = np.full((H, W), -1, dtype=np.int8)

        # Step 2: scan points → occupied + hard inflation
        ranges = np.array(self.scan.ranges)
        angles = self.scan.angle_min + np.arange(len(ranges)) * self.scan.angle_increment
        valid = np.isfinite(ranges) & (ranges > 0.0) & (ranges <= self.MAX_SCAN_RANGE)
        ranges = ranges[valid]
        angles = angles[valid]
        front = np.abs(angles) < self.SCAN_FOV
        ranges = ranges[front]
        angles = angles[front]

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        cols_pt = np.round(xs / res).astype(int) + self.GRID_REAR
        rows_pt = np.round(-ys / res).astype(int) + cy
        in_b = (rows_pt >= 0) & (rows_pt < H) & (cols_pt >= 0) & (cols_pt < W)

        # 끝점 마킹 + hard inflation (100)
        pad_h = self.hard_cells
        for r, c in zip(rows_pt[in_b], cols_pt[in_b]):
            r0 = max(r - pad_h, 0)
            r1 = min(r + pad_h + 1, H)
            c0 = max(c - pad_h, 0)
            c1 = min(c + pad_h + 1, W)
            kr0 = pad_h - (r - r0)
            kr1 = pad_h + (r1 - r)
            kc0 = pad_h - (c - c0)
            kc1 = pad_h + (c1 - c)
            k = self.hard_kernel[kr0:kr1, kc0:kc1]
            region = grid[r0:r1, c0:c1]
            region[k == 1] = 100

        # Step 3: 로봇 기준 정면 180도 레이캐스팅 (DDA) → 100 만날때까지 free(0)
        origin_r, origin_c = cy, self.GRID_REAR
        for ang in self.ray_angles:
            dx = math.cos(ang)
            dy = -math.sin(ang)  # row는 y 반전
            # DDA: 1셀씩 정밀 이동
            if abs(dx) > abs(dy):
                step = abs(dx)
            else:
                step = abs(dy)
            if step < 1e-9:
                continue
            dr = dy / step
            dc = dx / step
            r, c = float(origin_r), float(origin_c)
            for _ in range(self.ray_max_steps):
                r += dr
                c += dc
                ri, ci = int(round(r)), int(round(c))
                if ri < 0 or ri >= H or ci < 0 or ci >= W:
                    break
                if grid[ri, ci] == 100:
                    break
                grid[ri, ci] = 0  # free

        # Step 4: soft inflation — free(0) 영역만 정규분포 cost 확장
        pad_s = self.soft_cells
        occ_rows, occ_cols = np.where(grid == 100)
        for r, c in zip(occ_rows, occ_cols):
            r0 = max(r - pad_s, 0)
            r1 = min(r + pad_s + 1, H)
            c0 = max(c - pad_s, 0)
            c1 = min(c + pad_s + 1, W)
            kr0 = pad_s - (r - r0)
            kr1 = pad_s + (r1 - r)
            kc0 = pad_s - (c - c0)
            kc1 = pad_s + (c1 - c)
            k = self.soft_kernel[kr0:kr1, kc0:kc1]
            region = grid[r0:r1, c0:c1]
            can = (region >= 0) & (region < 100)
            region[can] = np.maximum(region[can], k[can])

        return grid

    # ── A* ─────────────────────────────────────────────────────

    def _astar(self, grid, start, goal):
        sr, sc = start
        gr, gc = goal

        if grid[gr, gc] == 100:
            orig_goal = (gr, gc)
            gr, gc = self._nearest_free(grid, gr, gc)
            if gr is None:
                self.get_logger().info(
                    f'A* abort: goal {orig_goal} is obstacle, nearest_free exhausted')
                return []
            self.get_logger().info(
                f'A* goal adjusted via nearest_free: {orig_goal} → ({gr},{gc}) '
                f'val={int(grid[gr, gc])}'
            )

        open_set = [(0.0, sr, sc)]
        came_from = {}
        g_score = {(sr, sc): 0.0}
        closed = set()
        neighbors_8 = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]
        sqrt2 = math.sqrt(2)

        iterations = 0
        while open_set:
            if iterations >= self.ASTAR_MAX_ITER:
                break
            iterations += 1
            _, cr, cc = heapq.heappop(open_set)

            if (cr, cc) in closed:
                continue
            closed.add((cr, cc))

            if cr == gr and cc == gc:
                path = [(cr, cc)]
                while (cr, cc) in came_from:
                    cr, cc = came_from[(cr, cc)]
                    path.append((cr, cc))
                path.reverse()
                return path

            for dr, dc in neighbors_8:
                nr, nc = cr + dr, cc + dc
                if nr < 0 or nr >= self.GRID_HEIGHT or nc < 0 or nc >= self.total_w:
                    continue
                if (nr, nc) in closed:
                    continue
                cell = int(grid[nr, nc])
                if cell == 100:
                    continue

                cost = sqrt2 if (dr != 0 and dc != 0) else 1.0
                if cell == -1:
                    cost *= self.UNKNOWN_COST
                elif cell > 0:
                    cost *= 1.0 + (cell / 99.0) * (self.INFLATION_COST_SCALE - 1.0)
                tg = g_score[(cr, cc)] + cost

                if tg < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = tg
                    came_from[(nr, nc)] = (cr, cc)
                    h = math.hypot(nr - gr, nc - gc)
                    heapq.heappush(open_set, (tg + h, nr, nc))

        return []

    def _nearest_free_start(self, grid, r, c, max_iter=300):
        """로봇 위치에서 BFS로 가장 가까운 free(<50) 셀 반환."""
        q = deque([(r, c)])
        visited = {(r, c)}
        count = 0
        while q and count < max_iter:
            count += 1
            cr, cc = q.popleft()
            val = int(grid[cr, cc])
            if 0 <= val < 50:
                return cr, cc
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.total_w:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
        return r, c  # fallback

    def _nearest_free(self, grid, r, c, max_iter=1000):
        q = deque([(r, c)])
        visited = {(r, c)}
        count = 0
        while q and count < max_iter:
            count += 1
            cr, cc = q.popleft()
            if grid[cr, cc] != 100:
                return cr, cc
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.total_w:
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
        self.get_logger().info(
            f'nearest_free: max_iter({max_iter}) exhausted from ({r},{c}), no free cell found')
        return None, None

    # def _farthest_free(self, grid, sr, sc):
    #     free_rows, free_cols = np.where((grid >= 0) & (grid < 50))
    #     if len(free_rows) == 0:
    #         return None
    #     dist_sq = (free_rows - sr) ** 2 + (free_cols - sc) ** 2
    #     idx = np.argmax(dist_sq)
    #     return int(free_rows[idx]), int(free_cols[idx])

    def _farthest_free(self, grid, sr, sc):
        free_rows, free_cols = np.where((grid >= 0) & (grid < 50))
        if len(free_rows) == 0:
            return None
        
        # 거리 * (주변 free cell 밀도) 로 scoring
        dist_sq = (free_rows - sr) ** 2 + (free_cols - sc) ** 2
        
        # 상위 20% 거리 후보 중에서 주변이 가장 열린 곳
        top_k = max(1, len(free_cols) // 5)
        top_idx = np.argpartition(dist_sq, -top_k)[-top_k:]
        
        # 각 후보 주변 5x5 free cell 수 계산
        best_score = -1
        best_idx = top_idx[0]
        for i in top_idx:
            r, c = free_rows[i], free_cols[i]
            r0, r1 = max(r-5,0), min(r+6, self.GRID_HEIGHT)
            c0, c1 = max(c-5,0), min(c+6, self.total_w)
            patch = grid[r0:r1, c0:c1]
            score = np.sum((patch >= 0) & (patch < 100))
            if score > best_score:
                best_score = score
                best_idx = i
        
        return int(free_rows[best_idx]), int(free_cols[best_idx])

    # ── Spline smoothing ───────────────────────────────────────

    def _spline_path(self, pts):
        """A* 경로를 numpy 자연 cubic spline으로 부드럽게 만들어 반환."""
        arr = np.array(pts, dtype=float)  # (N, 2)

        # SPLINE_SKIP 간격으로 제어점 선택 (첫점/끝점 포함)
        idx = list(range(0, len(arr), self.SPLINE_SKIP))
        if idx[-1] != len(arr) - 1:
            idx.append(len(arr) - 1)
        ctrl = arr[idx]  # (M, 2)
        n = len(ctrl)

        if n < 3:
            return [tuple(p) for p in arr]

        # 호 길이 기반 파라미터화
        h = np.linalg.norm(np.diff(ctrl, axis=0), axis=1)  # (M-1,)
        h = np.maximum(h, 1e-9)

        # 자연 cubic spline: 2계 미분 M을 연립방정식으로 풀기
        # 경계 조건: M[0] = M[-1] = 0 (자연 spline)
        A = np.zeros((n, n))
        rhs = np.zeros((n, 2))
        A[0, 0] = 1.0
        A[-1, -1] = 1.0
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i]     = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            rhs[i] = 3.0 * ((ctrl[i + 1] - ctrl[i]) / h[i]
                             - (ctrl[i] - ctrl[i - 1]) / h[i - 1])
        M = np.linalg.solve(A, rhs)  # (M, 2)

        # 각 세그먼트를 SPLINE_DENSITY개 샘플로 세분화
        dense = []
        K = self.SPLINE_DENSITY
        for i in range(n - 1):
            hi = h[i]
            s = np.linspace(0.0, 1.0, K, endpoint=False)
            a, b = 1.0 - s, s  # (K,)
            seg = (np.outer(a, ctrl[i]) + np.outer(b, ctrl[i + 1])
                   + (hi ** 2 / 6.0) * (np.outer(a ** 3 - a, M[i])
                                         + np.outer(b ** 3 - b, M[i + 1])))
            dense.extend(map(tuple, seg))
        dense.append(tuple(ctrl[-1]))
        return dense

    # ── Pure Pursuit ───────────────────────────────────────────

    def _pure_pursuit(self, path):
        lookahead_pt = None
        for x, y in path:
            if math.hypot(x, y) >= self.PP_LOOKAHEAD:
                lookahead_pt = (x, y)
                break
        if lookahead_pt is None:
            lookahead_pt = path[-1]

        tx, ty = lookahead_pt
        ld = math.hypot(tx, ty)
        if ld < 1e-6:
            return 0.0, lookahead_pt

        alpha = math.atan2(ty, tx)
        steer = math.atan2(2.0 * self.WHEELBASE * math.sin(alpha), ld)
        return steer, lookahead_pt

    # ── Coordinate transforms ─────────────────────────────────

    def _world_to_grid(self, x, y):
        cy = self.GRID_HEIGHT // 2
        col = int(round(x / self.GRID_RESOLUTION)) + self.GRID_REAR
        row = int(round(-y / self.GRID_RESOLUTION)) + cy
        return row, col

    def _grid_to_world(self, row, col):
        cy = self.GRID_HEIGHT // 2
        x = (col - self.GRID_REAR) * self.GRID_RESOLUTION
        y = -(row - cy) * self.GRID_RESOLUTION
        return x, y

    # ── Publishers ─────────────────────────────────────────────

    def _publish_grid(self, grid, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = 'laser'
        msg.info.resolution = float(self.GRID_RESOLUTION)
        msg.info.width = self.total_w
        msg.info.height = self.GRID_HEIGHT
        half_y = self.GRID_HEIGHT * self.GRID_RESOLUTION / 2.0
        msg.info.origin.position.x = -self.GRID_REAR * self.GRID_RESOLUTION
        msg.info.origin.position.y = -half_y
        msg.info.origin.position.z = 0.0
        flipped = np.flipud(grid)
        msg.data = flipped.flatten().tolist()
        self.grid_pub.publish(msg)

    def _publish_markers(self, path_world, lookahead_pt, stamp):
        ma = MarkerArray()

        # 1. Spline path (LINE_STRIP, 초록)
        path_m = Marker()
        path_m.header.stamp = stamp
        path_m.header.frame_id = 'laser'
        path_m.ns = 'astar_path'
        path_m.id = 0
        path_m.type = Marker.LINE_STRIP
        path_m.action = Marker.ADD
        path_m.pose.orientation.w = 1.0
        path_m.scale.x = self.MARKER_LINE_WIDTH
        path_m.color.r = 0.0
        path_m.color.g = 1.0
        path_m.color.b = 0.0
        path_m.color.a = 1.0
        path_m.points = [Point(x=float(x), y=float(y), z=0.0) for x, y in path_world]
        ma.markers.append(path_m)

        # 2. Lookahead 목표점 (SPHERE, 빨강)
        sphere = Marker()
        sphere.header.stamp = stamp
        sphere.header.frame_id = 'laser'
        sphere.ns = 'pp_lookahead'
        sphere.id = 1
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(lookahead_pt[0])
        sphere.pose.position.y = float(lookahead_pt[1])
        sphere.pose.position.z = 0.0
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = self.MARKER_SPHERE_SIZE
        sphere.scale.y = self.MARKER_SPHERE_SIZE
        sphere.scale.z = self.MARKER_SPHERE_SIZE
        sphere.color.r = 1.0
        sphere.color.g = 0.0
        sphere.color.b = 0.0
        sphere.color.a = 1.0
        ma.markers.append(sphere)

        # 3. 로봇→lookahead 연결선 (LINE_STRIP, 노랑)
        line = Marker()
        line.header.stamp = stamp
        line.header.frame_id = 'laser'
        line.ns = 'pp_line'
        line.id = 2
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = self.MARKER_LINE_WIDTH
        line.color.r = 1.0
        line.color.g = 1.0
        line.color.b = 0.0
        line.color.a = 1.0
        line.points = [Point(x=0.0, y=0.0, z=0.0),
                       Point(x=float(lookahead_pt[0]), y=float(lookahead_pt[1]), z=0.0)]
        ma.markers.append(line)

        self.marker_pub.publish(ma)


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
