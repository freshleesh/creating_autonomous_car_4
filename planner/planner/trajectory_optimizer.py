#!/usr/bin/env python3
import os
import csv
import math
import time
import numpy as np
from scipy.optimize import minimize

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory


class TrajectoryOptimizer(Node):

    PARAMS = {
        'safety_margin': 0.5,
        'v_max':         6.0,
        'a_lat_max':     6.0,
        'a_long_max':    4.0,
        'target_ds':     0.25,
    }

    def __init__(self):
        super().__init__('trajectory_optimizer')

        self.declare_parameter('map_name',   '')
        self.declare_parameter('input_csv',  'centerline.csv')
        self.declare_parameter('output_csv', 'global_waypoints.csv')
        for name, default in self.PARAMS.items():
            self.declare_parameter(name, default)

        p = lambda name: self.get_parameter(name).value

        map_name   = p('map_name')
        input_csv  = p('input_csv')
        output_csv = p('output_csv')

        if not map_name:
            self.get_logger().error('[TrajectoryOptimizer] map_name parameter is required!')
            return

        map_dir  = os.path.join(
            get_package_share_directory('stack_master'), 'maps', map_name)
        in_path  = os.path.join(map_dir, input_csv)
        out_path = os.path.join(map_dir, output_csv)

        x_c, y_c, w_r, w_l = self._load_centerline(in_path)
        self.get_logger().info(
            f'[TrajectoryOptimizer] Loaded {len(x_c)} centerline points from {in_path}')

        t0 = time.time()
        x, y, psi, kappa, vx, w_r_new, w_l_new = TrajectoryOptimizer._optimize(
            x_c, y_c, w_r, w_l,
            p('safety_margin'), p('v_max'), p('a_lat_max'), p('a_long_max'), p('target_ds'),
            map_dir=map_dir)
        elapsed = time.time() - t0

        self.get_logger().info(
            f'[TrajectoryOptimizer] Optimized in {elapsed:.2f}s, {len(x)} points, '
            f'success={x is not None}')

        self._save_global_waypoints(out_path, x, y, w_r_new, w_l_new, psi, kappa, vx)
        self.get_logger().info(f'[TrajectoryOptimizer] Saved to {out_path}')

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _load_centerline(csv_path):
        x, y, wr, wl = [], [], [], []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x.append(float(row['x_m']))
                y.append(float(row['y_m']))
                wr.append(float(row['w_tr_right_m']))
                wl.append(float(row['w_tr_left_m']))
        return np.array(x), np.array(y), np.array(wr), np.array(wl)

    @staticmethod
    def _resample_uniform(x, y, wr, wl, target_ds):
        """Resample a closed curve to uniform arc-length spacing."""
        dx = np.diff(x, append=x[0])
        dy = np.diff(y, append=y[0])
        seg = np.hypot(dx, dy)
        s = np.concatenate([[0.0], np.cumsum(seg[:-1])])

        n_pts = max(4, int(round((s[-1] + seg[-1]) / target_ds)))
        s_new = np.linspace(0.0, s[-1], n_pts, endpoint=False)

        x_r  = np.interp(s_new, s, x)
        y_r  = np.interp(s_new, s, y)
        wr_r = np.interp(s_new, s, wr)
        wl_r = np.interp(s_new, s, wl)
        return x_r, y_r, wr_r, wl_r

    @staticmethod
    def _geom(x, y):
        """Heading and curvature via central differences (closed path)."""
        dx1 = np.roll(x, -1) - np.roll(x, 1)
        dy1 = np.roll(y, -1) - np.roll(y, 1)
        dx2 = np.roll(x, -1) - 2.0 * x + np.roll(x, 1)
        dy2 = np.roll(y, -1) - 2.0 * y + np.roll(y, 1)
        psi   = np.arctan2(dy1, dx1)
        denom = np.maximum((dx1**2 + dy1**2)**1.5, 1e-9)
        kappa = (dx1 * dy2 - dy1 * dx2) / denom
        return psi, kappa

    @staticmethod
    def _speed_profile(kappa, v_max, a_lat_max, a_long_max, target_ds):
        """Velocity profile: lateral grip cap + forward-backward integration."""
        abs_k = np.maximum(np.abs(kappa), 1e-6)
        vx = np.minimum(v_max, np.sqrt(a_lat_max / abs_k))

        ds = target_ds
        # Forward pass
        for i in range(1, len(vx)):
            vx[i] = min(vx[i], math.sqrt(vx[i - 1]**2 + 2.0 * a_long_max * ds))
        # Two backward passes to close the loop
        for _ in range(2):
            for i in range(len(vx) - 1, -1, -1):
                j = (i + 1) % len(vx)
                vx[i] = min(vx[i], math.sqrt(vx[j]**2 + 2.0 * a_long_max * ds))

        return np.clip(vx, 0.1, v_max)

    @staticmethod
    def _save_global_waypoints(out_path, x, y, w_r, w_l, psi, kappa, vx):
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m',
                             'psi_rad', 'kappa_radpm', 'vx_mps'])
            for i in range(len(x)):
                writer.writerow([
                    f'{x[i]:.6f}',     f'{y[i]:.6f}',
                    f'{w_r[i]:.4f}',   f'{w_l[i]:.4f}',
                    f'{psi[i]:.6f}',   f'{kappa[i]:.6f}',
                    f'{vx[i]:.4f}',
                ])

    # ------------------------------------------------------------------ core

    @staticmethod
    def _optimize(x_c, y_c, w_r, w_l,
                  safety_margin, v_max, a_lat_max, a_long_max, target_ds,
                  map_dir=None, safety_margin_outer=None):
        """Min-curvature optimization — students implement.
        Returns (x, y, psi, kappa, vx, w_r_new, w_l_new) — all numpy arrays."""
        # TODO ← you implement
        # 1. Resample to uniform spacing
        x_r, y_r, wr_r, wl_r = TrajectoryOptimizer._resample_uniform(
            x_c, y_c, w_r, w_l, target_ds)
        N = len(x_r)

        # Use actual boundary CSVs for directional distances (right vs left)
        if map_dir is not None:
            from planner.track_bounds import TrackBounds
            tb = TrackBounds(map_dir)
            if tb.is_valid():
                wr_r, wl_r = tb.compute_distances(x_r, y_r)

        # 2. Left-pointing unit normals (90° CCW of forward tangent)
        dx = np.roll(x_r, -1) - np.roll(x_r, 1)
        dy = np.roll(y_r, -1) - np.roll(y_r, 1)
        mag = np.maximum(np.hypot(dx, dy), 1e-9)
        tx, ty = dx / mag, dy / mag   # unit tangent
        nx, ny = -ty,  tx             # unit left-normal (toward inner wall)

        # 3. Symmetric bounds: safety margin on both walls → allows true OIO
        lb = np.minimum(-(wr_r - safety_margin), 0.0)
        ub = np.maximum(  wl_r - safety_margin,  0.0)
        bounds = list(zip(lb, ub))

        # Corner alignment weight: force inward phase to coincide with actual corners
        _, kappa_cl = TrajectoryOptimizer._geom(x_r, y_r)
        kw = np.abs(kappa_cl)
        w_align = 0.3  # raise → more aggressive inward at corners; lower → purer min-curvature

        # 4. Objective + analytical gradient: min-curvature + corner alignment
        def _curve(alpha):
            px = x_r + alpha * nx;  py = y_r + alpha * ny
            u  = np.roll(px, -1) - np.roll(px, 1)
            v  = np.roll(py, -1) - np.roll(py, 1)
            a  = np.roll(px, -1) - 2.0*px + np.roll(px, 1)
            b  = np.roll(py, -1) - 2.0*py + np.roll(py, 1)
            D2  = u**2 + v**2
            D32 = np.maximum(D2**1.5, 1e-12)
            Ni  = u*b - v*a
            return u, v, a, b, D2, D32, Ni, Ni / D32

        def objective(alpha):
            *_, kap = _curve(alpha)
            return float(np.sum(kap**2)) - w_align * float(np.sum(kw * alpha))

        def gradient(alpha):
            u, v, a, b, D2, D32, Ni, kap = _curve(alpha)
            def _dk(U, V, A, B, Ni_, D2_, D32_, dU, dV, dA, dB):
                dN  = U*dB + dU*B - V*dA - dV*A
                dot = U*dU + V*dV
                return (dN - 3.0*Ni_*dot / np.maximum(D2_, 1e-12)) / D32_
            g = 2.0*kap * _dk(u, v, a, b, Ni, D2, D32,
                               0.0, 0.0, -2.0*nx, -2.0*ny)
            u1,v1,a1,b1 = np.roll(u,1),np.roll(v,1),np.roll(a,1),np.roll(b,1)
            N1,D21,D321,k1 = np.roll(Ni,1),np.roll(D2,1),np.roll(D32,1),np.roll(kap,1)
            g += 2.0*k1 * _dk(u1,v1,a1,b1, N1,D21,D321, nx,ny,nx,ny)
            un,vn,an,bn = np.roll(u,-1),np.roll(v,-1),np.roll(a,-1),np.roll(b,-1)
            Nn,D2n,D32n,kn = np.roll(Ni,-1),np.roll(D2,-1),np.roll(D32,-1),np.roll(kap,-1)
            g += 2.0*kn * _dk(un,vn,an,bn, Nn,D2n,D32n, -nx,-ny,nx,ny)
            return g - w_align * kw

        res = minimize(
            objective, np.zeros(N),
            method='L-BFGS-B', jac=gradient, bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8},
        )
        alpha = res.x

        # 5. Final raceline
        x_opt = x_r + alpha * nx
        y_opt = y_r + alpha * ny

        # Adjust remaining track widths at each point
        # positive alpha = moved LEFT → further from right wall, closer to left wall
        wr_new = np.maximum(0.05, wr_r + alpha)
        wl_new = np.maximum(0.05, wl_r - alpha)

        # 6. Geometry
        psi, kappa = TrajectoryOptimizer._geom(x_opt, y_opt)

        # 7. Velocity profile
        vx = TrajectoryOptimizer._speed_profile(
            kappa, v_max, a_lat_max, a_long_max, target_ds)

        return x_opt, y_opt, psi, kappa, vx, wr_new, wl_new


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryOptimizer()
    rclpy.spin_once(node, timeout_sec=2.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()