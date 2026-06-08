#!/usr/bin/env python3
"""
lap_analyzer.py  –  Lap timing + tracking-error logger for MPPI/PP evaluation.

Per completed lap it prints:
    * lap time              [s]
    * top speed             [m/s]
    * distance travelled    [m]
    * tracking error        ∫|d_cte| ds  [m^2]  (and its mean |d_cte| [m])

and continuously publishes the DRIVEN trajectory as a 3-D LINE_STRIP whose
HEIGHT (z) encodes speed — so corners (slow) sit low and straights (fast) rise
up. Colour also ramps blue→red with speed.

Lap detection: project ego onto the /global_waypoints polyline to get arc
length s; a lap is counted when s wraps (was near the end, now near the start)
after the car has passed the half-way point. The first crossing starts the
clock (the partial spawn→line segment is ignored).

Topics
------
Sub:  /vesc/odom (Odometry), /global_waypoints (WpntArray, latched)
Pub:  /lap_analyzer/trajectory (visualization_msgs/Marker, LINE_STRIP, map)
"""

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from f110_msgs.msg import WpntArray


class LapAnalyzer(Node):

    def __init__(self) -> None:
        super().__init__('lap_analyzer')

        gp = lambda name, val: self.declare_parameter(name, val).value
        self.odom_topic    = str(gp('odom_topic',   '/vesc/odom'))
        self.global_topic  = str(gp('global_topic', '/global_waypoints'))
        self.marker_topic  = str(gp('marker_topic', '/lap_analyzer/trajectory'))
        # cylinder height = speed * height_scale
        self.height_scale  = float(gp('height_scale', 0.1))
        self.cyl_diameter  = float(gp('cyl_diameter', 0.08))   # [m]
        # speed mapped to full blue->red colour ramp
        self.v_colour_max  = float(gp('v_colour_max', 8.0))   # [m/s]
        # only log a trajectory vertex once the car has moved this far
        self.sample_ds     = float(gp('sample_ds', 0.05))     # [m]

        # ---- reference ------------------------------------------------------
        self.ref_xy = None         # (N, 2) waypoints
        self.ref_s = None          # (N,) cumulative arc length
        self.s_total = 0.0

        # ---- per-lap state --------------------------------------------------
        self._reset_lap()
        self.lap_count = 0
        self.lap_started = False   # True once the start line is first crossed
        self.armed = False         # True once past half-way (enables wrap test)
        self.t0 = None             # lap start stamp [s]
        self.prev_s = None
        self.prev_xy = None

        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(WpntArray, self.global_topic, self._global_cb, latched)
        self.create_subscription(Odometry,  self.odom_topic,   self._odom_cb, 20)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 5)
        self.create_timer(0.2, self._publish_marker)   # 5 Hz viz

        self.get_logger().info(
            f'lap_analyzer up | odom={self.odom_topic} | '
            f'z=speed*{self.height_scale}')

    # ------------------------------------------------------------------ state
    def _reset_lap(self) -> None:
        self.traj = []             # list of (x, y, v)
        self.max_speed = 0.0
        self.err_integral = 0.0    # ∫|d_cte| ds  [m^2]
        self.dist = 0.0            # [m]

    # ------------------------------------------------------------------ subs
    def _global_cb(self, msg: WpntArray) -> None:
        if len(msg.wpnts) < 2:
            return
        xy = np.array([[w.x_m, w.y_m] for w in msg.wpnts], dtype=float)
        ds = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(ds)])
        loop_close = float(np.hypot(*(xy[0] - xy[-1])))
        self.ref_xy = xy
        self.ref_s = s
        self.s_total = float(s[-1] + loop_close)

    def _odom_cb(self, msg: Odometry) -> None:
        if self.ref_xy is None:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        v = float(msg.twist.twist.linear.x)
        t = self.get_clock().now().nanoseconds * 1e-9

        # nearest waypoint → cross-track error d and arc length s
        d2 = (self.ref_xy[:, 0] - x) ** 2 + (self.ref_xy[:, 1] - y) ** 2
        j = int(np.argmin(d2))
        d_cte = math.sqrt(float(d2[j]))
        s_now = float(self.ref_s[j])

        # accumulate distance + error integral
        if self.prev_xy is not None:
            ds = math.hypot(x - self.prev_xy[0], y - self.prev_xy[1])
            self.dist += ds
            self.err_integral += d_cte * ds
        self.prev_xy = (x, y)

        # trajectory sampling + max speed
        if not self.traj or math.hypot(x - self.traj[-1][0], y - self.traj[-1][1]) >= self.sample_ds:
            self.traj.append((x, y, v))
        self.max_speed = max(self.max_speed, abs(v))

        # lap detection via arc-length wrap
        half, near_end, near_start = 0.5 * self.s_total, 0.8 * self.s_total, 0.2 * self.s_total
        if s_now > half:
            self.armed = True
        if (self.armed and self.prev_s is not None
                and self.prev_s > near_end and s_now < near_start):
            self._on_lap_line(t)
            self.armed = False
        self.prev_s = s_now

    # ------------------------------------------------------------------ lap
    def _on_lap_line(self, t: float) -> None:
        if not self.lap_started:
            # first crossing = start the clock, discard the partial segment
            self.lap_started = True
            self.t0 = t
            self._reset_lap()
            self.get_logger().info('--- start line crossed, timing lap 1 ---')
            return

        self.lap_count += 1
        lap_time = t - self.t0
        mean_err = self.err_integral / max(self.dist, 1e-6)
        self.get_logger().info(
            '\n========== LAP %d ==========\n'
            '  time          : %.3f s\n'
            '  top speed      : %.2f m/s\n'
            '  distance       : %.2f m\n'
            '  tracking error : %.3f m^2  (mean |cte| %.3f m)\n'
            '============================'
            % (self.lap_count, lap_time, self.max_speed, self.dist,
               self.err_integral, mean_err))
        self.t0 = t
        self._reset_lap()

    # ------------------------------------------------------------------ viz
    def _publish_marker(self) -> None:
        if len(self.traj) < 1:
            return
        ma = MarkerArray()
        clear = Marker(); clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        stamp = self.get_clock().now().to_msg()
        vmax = max(self.v_colour_max, 1e-3)
        for i, (x, y, v) in enumerate(self.traj):
            h = max(abs(v) * self.height_scale, 1e-3)   # cylinder height = speed
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'driven_trajectory'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.5 * h          # base at ground, top at h
            m.pose.orientation.w = 1.0
            m.scale.x = self.cyl_diameter
            m.scale.y = self.cyl_diameter
            m.scale.z = h
            f = min(max(abs(v) / vmax, 0.0), 1.0)  # 0 slow → 1 fast
            m.color.r, m.color.g, m.color.b, m.color.a = f, 0.2, 1.0 - f, 1.0
            ma.markers.append(m)
        self.marker_pub.publish(ma)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LapAnalyzer()
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
