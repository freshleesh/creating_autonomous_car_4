#!/usr/bin/env python3
"""
detection_sh_v2.py  –  Reference-line proximity obstacle detector (sh, v2).

Difference from detection_sh.py: instead of subtracting the static map, this
keeps only scan points that lie CLOSE TO THE REFERENCE LINE (the raceline /
global waypoints). The idea: the line is where we drive, so anything sitting on
it is an obstacle we care about; wall returns are far from the line and dropped.

Pipeline:
    /scan + /odom + /global_waypoints
        1. LaserScan -> laser-frame (x, y)
        2. range gate (r_min .. r_max)
        3. transform each point to the map frame, keep it iff its distance to
           the nearest reference waypoint is <= ref_dist_thresh
        4. publish every survivor as an Obstacle (laser frame)
    -> /detections (f110_msgs/ObstacleArray)

Published x_m / y_m stay in the LASER frame (MPPI re-applies laser_x_offset +
ego pose). Keep `laser_to_base_x` equal to MPPI's `laser_x_offset`.

All tunables live in stack_master/config/detection_sh_v2.yaml.
"""

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from f110_msgs.msg import Obstacle, ObstacleArray, WpntArray


def _yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class DetectionSHv2(Node):

    def __init__(self) -> None:
        super().__init__('detection_sh_v2')

        gp = lambda name, val: self.declare_parameter(name, val).value

        # ---- topics ---------------------------------------------------------
        self.scan_topic       = str(gp('scan_topic',       '/scan'))
        self.odom_topic       = str(gp('odom_topic',       '/vesc/odom'))
        self.global_topic     = str(gp('global_topic',     '/global_waypoints'))
        self.centerline_topic = str(gp('centerline_topic', '/centerline_waypoints'))
        self.detections_topic = str(gp('detections_topic', '/detections'))
        self.markers_topic    = str(gp('markers_topic',    '/detection_sh/markers'))

        # ---- range gate -----------------------------------------------------
        self.r_min = float(gp('r_min', 0.05))   # [m]
        self.r_max = float(gp('r_max', 10.0))   # [m]

        # ---- reference-line proximity --------------------------------------
        # Keep a point iff its distance to the nearest reference waypoint is
        # <= this. So obstacles = scan returns sitting on/near the raceline.
        self.ref_dist_thresh = float(gp('ref_dist_thresh', 0.2))  # [m]
        # base_link -> laser x offset; MUST match MPPI's laser_x_offset.
        self.laser_to_base_x = float(gp('laser_to_base_x', 0.27))  # [m]

        # ---- output shaping -------------------------------------------------
        self.sort_by_range = bool(gp('sort_by_range', True))
        self.max_obstacles = int(gp('max_obstacles', 0))      # 0 = unlimited
        self.obstacle_size = float(gp('obstacle_size', 0.05)) # [m] reported size

        self.publish_markers = bool(gp('publish_markers', True))
        self.marker_size     = float(gp('marker_size', 0.3))  # [m] RViz sphere size

        # ---- state ----------------------------------------------------------
        self.have_pose = False
        self.ex = self.ey = self.eyaw = 0.0
        self.ref_xy = None          # (N, 2) reference polyline in map frame
        self._global_loaded = False

        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)

        self.create_subscription(WpntArray, self.global_topic,     self._global_cb, latched)
        self.create_subscription(WpntArray, self.centerline_topic, self._centerline_cb, latched)
        self.create_subscription(Odometry,  self.odom_topic, self._odom_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, 10)

        self.det_pub = self.create_publisher(ObstacleArray, self.detections_topic, 10)
        self.mark_pub = (self.create_publisher(MarkerArray, self.markers_topic, 5)
                         if self.publish_markers else None)

        self.get_logger().info(
            f'detection_sh_v2 up | scan={self.scan_topic} -> {self.detections_topic} | '
            f'ref_dist<={self.ref_dist_thresh}m | r=[{self.r_min},{self.r_max}]m')

    # ------------------------------------------------------------------ subs
    def _set_ref(self, msg: WpntArray) -> None:
        if len(msg.wpnts) < 2:
            return
        self.ref_xy = np.array([[w.x_m, w.y_m] for w in msg.wpnts], dtype=float)

    def _global_cb(self, msg: WpntArray) -> None:
        self._global_loaded = True
        self._set_ref(msg)

    def _centerline_cb(self, msg: WpntArray) -> None:
        if self._global_loaded:
            return
        self._set_ref(msg)

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.ex, self.ey = p.x, p.y
        self.eyaw = _yaw_from_quat(msg.pose.pose.orientation)
        self.have_pose = True

    # ------------------------------------------------------------------ scan
    def _scan_cb(self, msg: LaserScan) -> None:
        if not self.have_pose or self.ref_xy is None:
            return

        r = np.asarray(msg.ranges, dtype=float)
        n = r.shape[0]
        angles = msg.angle_min + np.arange(n) * msg.angle_increment

        valid = np.isfinite(r) & (r >= self.r_min) & (r <= self.r_max)
        xl = np.where(valid, r * np.cos(angles), np.nan)
        yl = np.where(valid, r * np.sin(angles), np.nan)

        # laser -> map frame
        c, s = math.cos(self.eyaw), math.sin(self.eyaw)
        bx = xl + self.laser_to_base_x
        by = yl
        gx = self.ex + c * bx - s * by
        gy = self.ey + s * bx + c * by

        # distance to nearest reference waypoint, only for valid points
        vidx = np.flatnonzero(valid)
        keep = np.zeros(n, dtype=bool)
        if vidx.size:
            dx = gx[vidx][:, None] - self.ref_xy[None, :, 0]
            dy = gy[vidx][:, None] - self.ref_xy[None, :, 1]
            dmin = np.sqrt(np.min(dx * dx + dy * dy, axis=1))
            keep[vidx] = dmin <= self.ref_dist_thresh

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

        if self.mark_pub is not None:
            self.mark_pub.publish(self._markers(scan, idx, gx, gy))

    def _markers(self, scan, idx, gx, gy) -> MarkerArray:
        ma = MarkerArray()
        clear = Marker(); clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = scan.header.stamp
        m.ns = 'detection_sh_v2'
        m.id = 0
        m.type = Marker.SPHERE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = max(self.marker_size, 0.05)
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 1.0, 0.8
        for i in idx:
            p = Point(); p.x = float(gx[i]); p.y = float(gy[i]); p.z = 0.1
            m.points.append(p)
        ma.markers.append(m)
        return ma


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectionSHv2()
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
