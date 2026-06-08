"""MPPI controller node — matches the I/O interface used by PP.py.

Topics
------
Sub:
    /vesc/odom         nav_msgs/Odometry          ego state
    /global_waypoints  f110_msgs/WpntArray        reference (latched)
    /detections        f110_msgs/ObstacleArray    (when use_detection=true)

Pub:
    /vesc/high_level/ackermann_cmd  ackermann_msgs/AckermannDriveStamped
    /mppi/reference                 visualization_msgs/Marker  (LINE_STRIP)
    /mppi/optimal_trajectory        visualization_msgs/Marker  (LINE_STRIP)
"""

import math
import os

# Force JAX onto CPU before importing JAX — the workspace machine ships JAX
# without CUDA, and we want predictable behavior either way.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from f110_msgs.msg import WpntArray, ObstacleArray

from controller.mppi.dynamics import (
    make_step,
    DEFAULT_STEER_VEL_SCALE,
    DEFAULT_ACCEL_SCALE,
    S_MIN, S_MAX, V_MIN, V_MAX,
)
from controller.mppi.mppi_core import MPPI


PARAMS = {
    # control loop / horizon
    'control_rate_hz':       20.0,
    'n_samples':            256,
    'n_steps':               12,
    'sim_time_step':          0.1,
    'n_iterations':           1,
    # sampling
    'control_std_steer':      0.5,
    'control_std_accel':      0.5,
    'steer_vel_scale':        DEFAULT_STEER_VEL_SCALE,
    'accel_scale':            DEFAULT_ACCEL_SCALE,
    # reference picking
    'ref_speed':              2.0,    # m/s, fallback when waypoint vx is 0
    'use_waypoint_speed':     True,
    'ref_speed_scale':        1.0,
    # Curvature speed cap on the reference: v_ref <= sqrt(max_lat_accel/|kappa|).
    # Single grip knob — lower it if the car slides in corners. 0 disables.
    'max_lat_accel':          8.0,    # m/s^2 (dry F1TENTH grip ~ mu*g ~ 10.3)
    # cost weights
    'xy_weight':              1.0,
    'velocity_weight':        0.1,
    'yaw_weight':             0.2,
    # MPPI hyperparams
    'temperature':            0.01,
    'damping':                0.001,
    # output clamps
    'max_steering_angle':     0.4,
    'min_speed':              0.0,
    'max_speed':              6.0,
    # detection (opp-aware) cost
    'use_detection':         False,
    'max_obstacles':           8,
    'obstacle_weight':        50.0,
    'obstacle_radius':         0.6,
    'laser_x_offset':         0.27,   # base_link → laser x [m]
    'detection_timeout':       0.5,
    # vehicle
    'wheelbase':              0.33,
}


def _yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class MPPINode(Node):

    def __init__(self):
        super().__init__('mppi')

        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        p = lambda name: self.get_parameter(name).value

        # Pull params once at startup (shapes drive JAX compilation).
        self.n_samples       = int(p('n_samples'))
        self.n_steps         = int(p('n_steps'))
        self.sim_dt          = float(p('sim_time_step'))
        self.n_iterations    = int(p('n_iterations'))
        self.max_obstacles   = int(p('max_obstacles'))

        self.use_detection   = bool(p('use_detection'))
        self.max_steer       = float(p('max_steering_angle'))
        self.min_speed       = float(p('min_speed'))
        self.max_speed       = float(p('max_speed'))
        self.laser_x_offset  = float(p('laser_x_offset'))
        self.detect_timeout  = float(p('detection_timeout'))

        # Build MPPI engine (compiles on first .update call).
        step_fn = make_step(self.sim_dt)
        control_std = [float(p('control_std_steer')), float(p('control_std_accel'))]
        norm_params = [float(p('steer_vel_scale')), float(p('accel_scale'))]
        self.mppi = MPPI(
            step_fn=step_fn,
            n_samples=self.n_samples,
            n_steps=self.n_steps,
            max_obs=self.max_obstacles,
            control_std=control_std,
            norm_params=norm_params,
        )

        # State
        self.odom = None
        self.waypoints = None        # np.ndarray (N, 4): [x, y, v, psi]
        self.waypoint_s = None       # cumulative arc length
        self.waypoint_kappa = None   # np.ndarray (N,): signed curvature [1/m]
        self.waypoint_total_len = 0.0
        self.last_drive_steer = 0.0
        self.last_drive_speed = 0.0
        self.last_detections_time = None
        self.obstacles_world = np.full((self.max_obstacles, 2), 1e6, dtype=np.float32)

        # Time-aware warm-start bookkeeping (decouples control rate from sim_dt).
        self._last_plan_time = None
        self._shift_accum = 0.0
        self.control_dt = 1.0 / float(p('control_rate_hz'))

        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.create_subscription(Odometry, '/vesc/odom', self._odom_cb, 10)
        # Primary reference (matches ppc.launch.xml): /global_waypoints from
        # trajectory_optimizer output. /centerline_waypoints serves as a
        # fallback when the optimizer hasn't been run for the map yet.
        self._global_loaded = False
        self.create_subscription(WpntArray, '/global_waypoints', self._global_wp_cb, latched)
        self.create_subscription(WpntArray, '/centerline_waypoints', self._centerline_wp_cb, latched)
        if self.use_detection:
            self.create_subscription(ObstacleArray, '/detections', self._det_cb, 10)
            self.get_logger().info('Detection enabled — opponent keep-out cost active')
        else:
            self.get_logger().info('Detection disabled — opponent cost off')

        self.drive_pub     = self.create_publisher(AckermannDriveStamped, '/vesc/high_level/ackermann_cmd', 10)
        self.ref_pub       = self.create_publisher(Marker, '/mppi/reference', 10)
        self.traj_pub      = self.create_publisher(Marker, '/mppi/optimal_trajectory', 10)

        self.create_timer(1.0 / float(p('control_rate_hz')), self._loop)
        self.get_logger().info(
            f'MPPINode ready (S={self.n_samples}, T={self.n_steps}, dt={self.sim_dt:.2f}s, '
            f'use_detection={self.use_detection})'
        )

    # ------------------------------------------------------------------ subs
    def _odom_cb(self, msg):
        self.odom = msg

    def _load_wpnts(self, msg, source):
        N = len(msg.wpnts)
        if N < 2:
            return
        # Skip re-load when waypoint_publisher just republishes the latched msg
        # (same source, same count) — only count changes or a source upgrade
        # (centerline -> global) deserve the log line.
        if (self.waypoints is not None
                and self.waypoints.shape[0] == N
                and source == getattr(self, '_waypoint_source', None)):
            return
        arr = np.zeros((N, 4), dtype=np.float32)
        for i, w in enumerate(msg.wpnts):
            arr[i, 0] = w.x_m
            arr[i, 1] = w.y_m
            arr[i, 2] = w.vx_mps
            arr[i, 3] = w.psi_rad
        diffs = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(diffs)])
        loop_close = float(np.linalg.norm(arr[0, :2] - arr[-1, :2]))
        # Signed curvature via 3-point finite differences (closed-loop wrap).
        x, y = arr[:, 0], arr[:, 1]
        xn, xp = np.roll(x, -1), np.roll(x, 1)
        yn, yp = np.roll(y, -1), np.roll(y, 1)
        dx, dy = (xn - xp) * 0.5, (yn - yp) * 0.5
        ddx, ddy = xn - 2 * x + xp, yn - 2 * y + yp
        denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
        kappa = (dx * ddy - dy * ddx) / denom
        self.waypoints = arr
        self.waypoint_s = s.astype(np.float32)
        self.waypoint_kappa = kappa.astype(np.float32)
        self.waypoint_total_len = float(s[-1] + loop_close)
        self._waypoint_source = source
        self.get_logger().info(
            f'[MPPI] Loaded {N} waypoints from {source}, total ~{self.waypoint_total_len:.1f} m'
        )

    def _global_wp_cb(self, msg):
        self._global_loaded = True
        self._load_wpnts(msg, source='/global_waypoints')

    def _centerline_wp_cb(self, msg):
        # Only used if /global_waypoints hasn't arrived (matches PP's strict
        # preference for the optimized raceline when available).
        if self._global_loaded:
            return
        self._load_wpnts(msg, source='/centerline_waypoints (fallback)')

    def _det_cb(self, msg):
        if not self.use_detection or self.odom is None:
            return
        # Transform laser-frame obstacle centers to map frame using ego pose +
        # a static laser_x_offset along base_link x. (Sim and real both place
        # the LiDAR roughly 0.27 m forward of base_link; tunable via param.)
        ex = self.odom.pose.pose.position.x
        ey = self.odom.pose.pose.position.y
        yaw = _yaw_from_quat(self.odom.pose.pose.orientation)
        c, s = math.cos(yaw), math.sin(yaw)

        pts = np.full((self.max_obstacles, 2), 1e6, dtype=np.float32)
        n = min(self.max_obstacles, len(msg.obstacles))
        for i in range(n):
            o = msg.obstacles[i]
            # laser frame → base_link frame: shift +laser_x_offset along x
            bx = o.x_m + self.laser_x_offset
            by = o.y_m
            # base_link → map
            pts[i, 0] = ex + c * bx - s * by
            pts[i, 1] = ey + s * bx + c * by
        self.obstacles_world = pts
        self.last_detections_time = self.get_clock().now()

    # --------------------------------------------------------------- helpers
    def _closest_waypoint_idx(self, x, y):
        d2 = (self.waypoints[:, 0] - x) ** 2 + (self.waypoints[:, 1] - y) ** 2
        return int(np.argmin(d2))

    def _build_reference(self, x, y):
        """Pick n_steps waypoints by walking along arc length at ref speed."""
        idx = self._closest_waypoint_idx(x, y)
        s0 = float(self.waypoint_s[idx])
        N = self.waypoints.shape[0]
        ref_v_param = float(self.get_parameter('ref_speed').value)
        use_wp_v = bool(self.get_parameter('use_waypoint_speed').value)
        v_scale = float(self.get_parameter('ref_speed_scale').value)
        max_lat = float(self.get_parameter('max_lat_accel').value)

        ref = np.zeros((self.n_steps, 4), dtype=np.float32)
        s = s0
        # Initial speed for stepping: use waypoint or fallback.
        v0 = self.waypoints[idx, 2]
        v_step = float(v0 if (use_wp_v and v0 > 1e-3) else ref_v_param)
        for t in range(self.n_steps):
            v_step = max(0.1, v_step * v_scale if t == 0 else v_step)
            s = (s + v_step * self.sim_dt) % self.waypoint_total_len
            # Find segment index by binary search on cumulative arc length.
            j = int(np.searchsorted(self.waypoint_s, s, side='right')) - 1
            j = max(0, min(j, N - 1))
            wp_v = float(self.waypoints[j, 2])
            v_ref = (wp_v if (use_wp_v and wp_v > 1e-3) else ref_v_param) * v_scale
            # Curvature speed cap: keep lateral accel v^2*|kappa| <= max_lat_accel
            # so MPPI tracks a slower corner target and brakes ahead of it.
            kap = abs(float(self.waypoint_kappa[j]))
            if max_lat > 0.0 and kap > 1e-4:
                v_ref = min(v_ref, math.sqrt(max_lat / kap))
            v_step = v_ref
            ref[t, 0] = self.waypoints[j, 0]
            ref[t, 1] = self.waypoints[j, 1]
            ref[t, 2] = v_ref
            ref[t, 3] = self.waypoints[j, 3]
        return ref

    def _obstacles_for_solve(self):
        if not self.use_detection or self.last_detections_time is None:
            return np.full((self.max_obstacles, 2), 1e6, dtype=np.float32)
        age = (self.get_clock().now() - self.last_detections_time).nanoseconds * 1e-9
        if age > self.detect_timeout:
            return np.full((self.max_obstacles, 2), 1e6, dtype=np.float32)
        return self.obstacles_world

    def _publish_line_strip(self, pub, xy, ns, rgb, width=0.05):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.ns = ns
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = width
        m.color.r, m.color.g, m.color.b, m.color.a = rgb[0], rgb[1], rgb[2], 1.0
        for x, y in xy:
            p = Point(); p.x = float(x); p.y = float(y); p.z = 0.0
            m.points.append(p)
        pub.publish(m)

    # ------------------------------------------------------------------ loop
    def _loop(self):
        if self.odom is None or self.waypoints is None:
            return

        # Ego state from odom (gym_bridge fills twist with map-frame velocity).
        ex = self.odom.pose.pose.position.x
        ey = self.odom.pose.pose.position.y
        yaw = _yaw_from_quat(self.odom.pose.pose.orientation)
        # twist is in the BODY frame: gym_bridge fills linear.x = longitudinal
        # speed and linear.y = 0 (f110_gym base_classes: linear_vels_x = state
        # velocity, linear_vels_y = 0). The EKF-filtered /car_state/odom on the
        # real car likewise reports body-frame twist. So forward speed is
        # linear.x directly — do NOT project by yaw (that scaled it by cos(yaw),
        # collapsing v to ~0 / negative as the car turned around the loop).
        v = self.odom.twist.twist.linear.x

        x0 = np.array([ex, ey, self.last_drive_steer, v, yaw], dtype=np.float32)

        reference = self._build_reference(ex, ey)
        obstacles = self._obstacles_for_solve()

        weights = np.array([
            float(self.get_parameter('xy_weight').value),
            float(self.get_parameter('velocity_weight').value),
            float(self.get_parameter('yaw_weight').value),
            float(self.get_parameter('obstacle_weight').value) if self.use_detection else 0.0,
            float(self.get_parameter('obstacle_radius').value),
        ], dtype=np.float32)

        temperature = float(self.get_parameter('temperature').value)
        damping = float(self.get_parameter('damping').value)

        # Real time elapsed since the last solve → how many sim_dt prediction
        # steps to shift the warm-start by (0 most cycles at 50 Hz/sim_dt=0.05).
        now = self.get_clock().now()
        if self._last_plan_time is None:
            dt_real = self.control_dt
        else:
            dt_real = (now - self._last_plan_time).nanoseconds * 1e-9
        self._last_plan_time = now
        dt_real = float(np.clip(dt_real, 0.0, 5.0 * self.control_dt))
        self._shift_accum += dt_real
        n_shift = int(self._shift_accum / self.sim_dt)
        self._shift_accum -= n_shift * self.sim_dt

        a_opt, traj_opt = self.mppi.update(
            x0, reference, obstacles, weights,
            temperature=temperature, damping=damping, n_iter=self.n_iterations,
            n_shift=n_shift,
        )

        # First action in normalized units → physical units → next-step state.
        u0_norm = np.array(a_opt[0])
        steer_vel = float(u0_norm[0]) * float(self.get_parameter('steer_vel_scale').value)
        accel = float(u0_norm[1]) * float(self.get_parameter('accel_scale').value)

        # Command: speed/steering setpoint at the end of the first prediction
        # step. Projected over sim_dt (NOT the control period): the low-level
        # speed PID accelerates proportional to (cmd_speed - v), so shrinking
        # this dt to 1/control_rate starved acceleration ~2.5x and, via the
        # slow car vs fast reference mismatch, caused hard corner cut-in.
        cmd_steer = float(np.clip(self.last_drive_steer + steer_vel * self.sim_dt,
                                  -self.max_steer, self.max_steer))
        cmd_speed = float(np.clip(v + accel * self.sim_dt, self.min_speed, self.max_speed))

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = cmd_steer
        msg.drive.speed = cmd_speed
        self.drive_pub.publish(msg)

        self.last_drive_steer = cmd_steer
        self.last_drive_speed = cmd_speed

        # Visualization
        if traj_opt is not None:
            traj_np = np.array(traj_opt)
            self._publish_line_strip(self.traj_pub, traj_np[:, :2],
                                     ns='mppi_optimal', rgb=(0.1, 1.0, 0.2), width=0.08)
        self._publish_line_strip(self.ref_pub, reference[:, :2],
                                 ns='mppi_reference', rgb=(0.2, 0.4, 1.0), width=0.06)


def main(args=None):
    rclpy.init(args=args)
    node = MPPINode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
