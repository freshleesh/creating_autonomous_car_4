#!/usr/bin/env python3
"""
detection_sh.py  –  Minimal map-subtraction obstacle detector (sh).

Pipeline (no clustering / no fitting — every surviving point is an obstacle):
    /scan + /odom + /map
        1. LaserScan -> laser-frame (x, y)
        2. range gate (r_min .. r_max)
        3. wall subtraction: transform each point to the map frame and drop
           any point that lands within `wall_clear_radius` of an occupied
           OccupancyGrid cell
        4. publish every remaining point as an Obstacle (laser frame)
    -> /detections (f110_msgs/ObstacleArray)

The published x_m / y_m stay in the LASER frame, which is exactly what the
MPPI node expects on /detections: it re-applies laser_x_offset + ego pose to
put them back in the map frame. Keep `laser_to_base_x` here equal to MPPI's
`laser_x_offset` so the wall check and MPPI agree on where each point is.

All tunables live in stack_master/config/detection_sh.yaml.
"""

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from f110_msgs.msg import Obstacle, ObstacleArray


def _yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class DetectionSH(Node):

    def __init__(self) -> None:
        super().__init__('detection_sh')

        gp = lambda name, val: self.declare_parameter(name, val).value

        # ---- topics ---------------------------------------------------------
        self.scan_topic       = str(gp('scan_topic',       '/scan'))
        self.odom_topic       = str(gp('odom_topic',       '/vesc/odom'))
        self.map_topic        = str(gp('map_topic',        '/map'))
        self.detections_topic = str(gp('detections_topic', '/detections'))
        self.markers_topic    = str(gp('markers_topic',    '/detection_sh/markers'))

        # ---- range gate -----------------------------------------------------
        self.r_min = float(gp('r_min', 0.05))   # [m] drop returns closer than this
        self.r_max = float(gp('r_max', 10.0))   # [m] drop returns farther than this

        # ---- wall subtraction ----------------------------------------------
        # Drop a scan point if a map cell within this radius is occupied.
        self.wall_clear_radius  = float(gp('wall_clear_radius', 0.01))  # [m]
        self.occupied_threshold = int(gp('occupied_threshold', 50))     # 0..100
        # base_link -> laser x offset; MUST match MPPI's laser_x_offset so the
        # wall check transforms points to the same map location MPPI will.
        self.laser_to_base_x = float(gp('laser_to_base_x', 0.27))       # [m]

        # ---- output shaping -------------------------------------------------
        # Sort survivors nearest-first so that, when a downstream consumer caps
        # the list (MPPI keeps only its first max_obstacles), the closest points
        # are the ones kept. Set max_obstacles>0 to also cap here.
        self.sort_by_range = bool(gp('sort_by_range', True))
        self.max_obstacles = int(gp('max_obstacles', 0))   # 0 = unlimited
        self.obstacle_size = float(gp('obstacle_size', 0.05))  # [m] reported size

        self.publish_markers = bool(gp('publish_markers', True))

        # ---- state ----------------------------------------------------------
        self.have_pose = False
        self.ex = self.ey = self.eyaw = 0.0
        self._map = None        # raw OccupancyGrid
        self._map_arr = None    # np.int8 (H, W)
        self._res = self._ox = self._oy = 0.0
        self._mh = self._mw = 0

        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)

        self.create_subscription(OccupancyGrid, self.map_topic,  self._map_cb, latched)
        self.create_subscription(Odometry,      self.odom_topic, self._odom_cb, 10)
        self.create_subscription(LaserScan,     self.scan_topic, self._scan_cb, 10)

        self.det_pub = self.create_publisher(ObstacleArray, self.detections_topic, 10)
        self.mark_pub = (self.create_publisher(MarkerArray, self.markers_topic, 5)
                         if self.publish_markers else None)

        self.get_logger().info(
            f'detection_sh up | scan={self.scan_topic} -> {self.detections_topic} | '
            f'r=[{self.r_min},{self.r_max}]m | wall_clear={self.wall_clear_radius}m')

    # ------------------------------------------------------------------ subs
    def _map_cb(self, msg: OccupancyGrid) -> None:
        self._map = msg
        info = msg.info
        self._map_arr = np.array(msg.data, dtype=np.int8).reshape(info.height, info.width)
        self._res = info.resolution
        self._ox = info.origin.position.x
        self._oy = info.origin.position.y
        self._mh = info.height
        self._mw = info.width

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.ex, self.ey = p.x, p.y
        self.eyaw = _yaw_from_quat(msg.pose.pose.orientation)
        self.have_pose = True

    # ------------------------------------------------------------------ scan
    def _scan_cb(self, msg: LaserScan) -> None:
        r = np.asarray(msg.ranges, dtype=float)
        n = r.shape[0]
        angles = msg.angle_min + np.arange(n) * msg.angle_increment

        valid = np.isfinite(r) & (r >= self.r_min) & (r <= self.r_max)
        xl = np.where(valid, r * np.cos(angles), np.nan)
        yl = np.where(valid, r * np.sin(angles), np.nan)

        # --- wall subtraction (needs pose + map) ---------------------------
        keep = valid.copy()
        gx = gy = None
        if self.have_pose and self._map_arr is not None:
            c, s = math.cos(self.eyaw), math.sin(self.eyaw)
            bx = xl + self.laser_to_base_x
            by = yl
            gx = self.ex + c * bx - s * by
            gy = self.ey + s * bx + c * by

            cols = np.where(valid, (gx - self._ox) / self._res, -1).astype(int)
            rows = np.where(valid, (gy - self._oy) / self._res, -1).astype(int)
            in_b = valid & (cols >= 0) & (cols < self._mw) & (rows >= 0) & (rows < self._mh)

            cr = int(round(self.wall_clear_radius / self._res)) if self._res > 0 else 0
            on_wall = np.zeros(n, dtype=bool)
            for dr in range(-cr, cr + 1):
                for dc in range(-cr, cr + 1):
                    r2 = np.clip(rows + dr, 0, self._mh - 1)
                    c2 = np.clip(cols + dc, 0, self._mw - 1)
                    on_wall |= in_b & (self._map_arr[r2, c2] >= self.occupied_threshold)
            keep &= ~on_wall

        idx = np.flatnonzero(keep)
        if self.sort_by_range and idx.size:
            idx = idx[np.argsort(r[idx])]   # nearest first
        if self.max_obstacles > 0:
            idx = idx[:self.max_obstacles]

        self._publish(msg, idx, xl, yl, gx, gy)

    # ----------------------------------------------------------------- pub
    def _publish(self, scan, idx, xl, yl, gx, gy) -> None:
        arr = ObstacleArray()
        arr.header.stamp = scan.header.stamp
        arr.header.frame_id = scan.header.frame_id   # laser frame
        for oid, i in enumerate(idx):
            o = Obstacle()
            o.id = int(oid)
            o.x_m = float(xl[i])
            o.y_m = float(yl[i])
            o.size = float(self.obstacle_size)
            o.is_visible = True
            o.is_static = False
            arr.obstacles.append(o)
        self.det_pub.publish(arr)

        if self.mark_pub is not None and gx is not None:
            self.mark_pub.publish(self._markers(scan, idx, gx, gy))

    def _markers(self, scan, idx, gx, gy) -> MarkerArray:
        ma = MarkerArray()
        clear = Marker(); clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = scan.header.stamp
        m.ns = 'detection_sh'
        m.id = 0
        m.type = Marker.SPHERE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = max(self.obstacle_size, 0.05)
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 1.0, 0.8
        for i in idx:
            p = Point(); p.x = float(gx[i]); p.y = float(gy[i]); p.z = 0.1
            m.points.append(p)
        ma.markers.append(m)
        return ma


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectionSH()
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
