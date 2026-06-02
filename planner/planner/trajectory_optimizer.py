#!/usr/bin/env python3

import os
import csv
import numpy as np
import osqp
from scipy import sparse

import rclpy
from rclpy.node import Node


class TrajectoryOptimizer(Node):

    def __init__(self):
        super().__init__('trajectory_optimizer')

        # ---- Parameters --------------------------------------------------
        self.declare_parameter('map_name', '')
        self.declare_parameter('input_csv', 'centerline.csv')
        self.declare_parameter('output_csv', 'global_waypoints.csv')
        self.declare_parameter('safety_margin', 0.20)   # [m]   clearance from each wall
        self.declare_parameter('v_max',         6.0)    # [m/s] vehicle speed cap
        self.declare_parameter('a_lat_max',     6.0)    # [m/s^2] lateral grip limit
        self.declare_parameter('a_long_max',    4.0)    # [m/s^2] longitudinal accel limit
        self.declare_parameter('target_ds',     0.25)   # [m]   uniform ds for QP input/output

        map_name      = self.get_parameter('map_name').value
        input_csv     = self.get_parameter('input_csv').value
        output_csv    = self.get_parameter('output_csv').value
        safety_margin = self.get_parameter('safety_margin').value
        v_max         = self.get_parameter('v_max').value
        a_lat_max     = self.get_parameter('a_lat_max').value
        a_long_max    = self.get_parameter('a_long_max').value
        target_ds     = self.get_parameter('target_ds').value

        if not map_name:
            self.get_logger().error('[TrajectoryOptimizer] map_name parameter is required!')
            return

        # ---- I/O paths ---------------------------------------------------
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        map_dir  = os.path.join(pkg_root, 'stack_master', 'maps', map_name)
        in_path  = os.path.join(map_dir, input_csv)
        out_path = os.path.join(map_dir, output_csv)

        if not os.path.exists(in_path):
            self.get_logger().error(f'[TrajectoryOptimizer] input not found: {in_path}')
            return

        # ---- Load + optimize + save -------------------------------------
        self.get_logger().info(f'[TrajectoryOptimizer] loading: {in_path}')
        x_c, y_c, w_r, w_l = self._load_centerline(in_path)

        self.get_logger().info(
            f'[TrajectoryOptimizer] optimizing on {len(x_c)} centerline points '
            f'(margin={safety_margin}, v_max={v_max}, a_lat={a_lat_max}, '
            f'a_long={a_long_max}, target_ds={target_ds})'
        )
        x_opt, y_opt, psi, kappa, vx, w_r_new, w_l_new = self._optimize(
            x_c, y_c, w_r, w_l,
            safety_margin=safety_margin,
            v_max=v_max,
            a_lat_max=a_lat_max,
            a_long_max=a_long_max,
            target_ds=target_ds,
        )

        self._save_global_waypoints(out_path, x_opt, y_opt, w_r_new, w_l_new, psi, kappa, vx)
        self.get_logger().info(
            f'[TrajectoryOptimizer] saved {len(x_opt)} pts → {out_path} '
            f'(v_min={vx.min():.2f}, v_max={vx.max():.2f} m/s, '
            f'|kappa|max={np.max(np.abs(kappa)):.3f})'
        )

    # ======================================================================
    #                       STUDENT IMPLEMENTATION
    # ======================================================================
    @staticmethod
    def _optimize(x_c, y_c, w_r, w_l,
                  safety_margin, v_max, a_lat_max, a_long_max, target_ds):
        """
        Minimum-curvature trajectory optimization.

        Parameters
        ----------
        x_c, y_c   : (N,) centerline coordinates (closed loop, no duplicate end)
        w_r, w_l   : (N,) track half-widths to the right / left walls
        safety_margin : [m] keep this far from each wall
        v_max, a_lat_max, a_long_max : vehicle limits
        target_ds  : [m] desired arc-length spacing for the optimized output

        Returns
        -------
        x_opt, y_opt, psi, kappa, vx : (M,) arrays for the optimized raceline
        w_r_new, w_l_new             : (M,) remaining clearance to the original walls
        """

        # TODO: Minimum-Curvature Path Optimization
        #
        # ┌─ Step 1. (optional) resample the centerline to uniform ds
        # │   x_r, y_r, w_r_r, w_l_r = _resample_uniform(x_c, y_c, w_r, w_l, target_ds)
        # │
        # ├─ Step 2. compute the unit normal vector n_i at every point.
        # │   tangent t_i via centered difference → normalize →
        # │   normal n_i = (-t_y, t_x) (pointing left)
        # │
        # ├─ Step 3. formulate the optimization problem (see lecture notes).
        # │   raceline parametrization:  r_i = p_i + alpha_i * n_i
        # │   objective :  minimize total curvature, e.g. min Sum kappa_i^2
        # │   bound     :  alpha_i in [ -w_r_i + safety_margin, +w_l_i - safety_margin ]
        # │   You may approximate curvature however you like (finite differences,
        # │   spline derivatives, etc.) and choose a different objective if you
        # │   prefer (shortest path, min lap time, etc. — see lecture).
        # │
        # ├─ Step 4. solve it.
        # │   Any optimization tool is fair game — convex QP solvers (osqp, cvxpy,
        # │   quadprog), general nonlinear minimizers (scipy.optimize.minimize),
        # │   evolution strategies (CMA-ES), or a ready-made package such as
        # │   trajectory_planning_helpers. Pick whichever fits your formulation.
        # │
        # ├─ Step 5. recover the optimal line coordinates from alpha.
        # │   x_opt = x_r + nx * alpha,   y_opt = y_r + ny * alpha
        # │
        # ├─ Step 6. (recommended) dedupe + cubic-spline resample to uniform ds
        # │   prevents clustered points on the inside of corners → clean psi, kappa.
        # │
        # ├─ Step 7. compute psi (heading) and kappa (curvature) via _geom()
        # │   kappa = (x' y'' − y' x'') / (x'^2 + y'^2)^(3/2)         (slide 45)
        # │
        # ├─ Step 8. speed profile via _speed_profile()                (slide 47)
        # │   - cornering limit:  v_max = sqrt(a_y / kappa)
        # │   - forward pass (accel limit) + backward pass (brake limit)
        # │   - pointwise min(cornering, accel, brake)
        # │
        # └─ Step 9. compute remaining clearance to the original walls (w_r_new, w_l_new)
        #             and return it together with the optimized line.

        # --- Step 1. uniform resample of the centerline --------------------
        x_r, y_r, w_r_r, w_l_r = TrajectoryOptimizer._resample_uniform(
            x_c, y_c, w_r, w_l, target_ds)
        N = len(x_r)

        # --- Step 2. helper: unit (left-pointing) normal at any line -------
        def _normals(xs, ys):
            dxn = (np.roll(xs, -1) - np.roll(xs, 1)) * 0.5
            dyn = (np.roll(ys, -1) - np.roll(ys, 1)) * 0.5
            Ln  = np.hypot(dxn, dyn) + 1e-12
            return -dyn / Ln, dxn / Ln

        # Smoothness operator on alpha (2nd difference) — small weight so the
        # wide-tight-wide racing-line shape is preserved but jitter is removed.
        S = np.zeros((N, N))
        for i in range(N):
            im1, ip1 = (i - 1) % N, (i + 1) % N
            S[i, im1] =  1.0
            S[i, i  ] = -2.0
            S[i, ip1] =  1.0
        StS = S.T @ S
        # Racer-like aggressive line:
        #   lambda_smooth (penalty on alpha's 2nd difference) is the force
        #   that pulls the line toward the centerline. When the track is
        #   wide, the centerline is far from both walls, so even a moderate
        #   lambda_smooth keeps the line stranded in the middle and prevents
        #   the apex dip a real racer would take.
        #   We push this VERY low during racing so the line is essentially
        #   only restrained by the inner/outer wall bounds, then crank it
        #   up x10 in polishing to remove any zig-zag the aggressive line
        #   left behind.
        lambda_smooth = 0.03      # very loose — let the line fully dive to the apex

        # Curvature-jerk operator on alpha (4th difference). alpha's 4th diff
        # is proportional to kappa's 2nd diff, i.e. the rate-of-change of
        # curvature. Penalising it kills the sharp curvature spikes that
        # appear at corner entry/exit when the racing line snaps inward.
        T = np.zeros((N, N))
        for i in range(N):
            im2, im1, ip1, ip2 = (i - 2) % N, (i - 1) % N, (i + 1) % N, (i + 2) % N
            T[i, im2] =  1.0
            T[i, im1] = -4.0
            T[i, i  ] =  6.0
            T[i, ip1] = -4.0
            T[i, ip2] =  1.0
        TtT = T.T @ T
        lambda_jerk = 0.5

        # Shortest-path term (Option D).
        # L² = Σ|Δr|², quadratic in da → adds to H and g as λ_short · L².
        lambda_short = 0.08

        # Asymmetric wall margins (curvature-aware apex room).
        # --------------------------------------------------
        # Plain symmetric bounds give the same margin to inner and outer
        # walls everywhere. In wide corners that means the line can hug
        # neither — it has equal pull from both sides and settles in
        # between. What a real racer wants is room to dive deep on the
        # INSIDE of every corner while still respecting wall safety on
        # the OUTSIDE.
        #
        # We give the line:
        #   - small margin (margin_inner) on the inside-of-the-corner wall
        #   - large margin (margin_outer) on the outside-of-the-corner wall
        #   - smooth blend at the centre / straights (avg of the two)
        # 'Inner' is decided by the SIGN of the centerline curvature κ_c:
        #   κ_c > 0  → centerline turns LEFT  → inner wall is the LEFT one
        #   κ_c < 0  → centerline turns RIGHT → inner wall is the RIGHT one
        #
        # The corner-vs-straight weight is |κ_c| smoothed over a short
        # window so the transition into / out of a corner is gradual.
        kappa_c = TrajectoryOptimizer._geom(x_r, y_r)[1]
        kappa_mag = np.abs(kappa_c)
        # short Gaussian smooth so the margin schedule is continuous
        sigma_kw = 4
        kw_kern = np.exp(-np.arange(-3*sigma_kw, 3*sigma_kw + 1) ** 2 / (2 * sigma_kw ** 2))
        kw_kern /= kw_kern.sum()
        kmag_s = np.zeros_like(kappa_mag)
        for j in range(-3 * sigma_kw, 3 * sigma_kw + 1):
            kmag_s += kw_kern[j + 3 * sigma_kw] * np.roll(kappa_mag, -j)
        turn = np.clip(kmag_s / max(kmag_s.max(), 1e-6), 0.0, 1.0)   # 0..1
        ksign = np.sign(np.where(np.abs(kappa_c) > 1e-3, kappa_c,
                                  np.roll(kappa_c, -1) + np.roll(kappa_c, 1)))

        # Apex still gets a tighter margin than the outer side, but not
        # razor-thin — 5 cm felt too risky in sim. 10 cm at the apex still
        # gives noticeably more "inside" room than a symmetric layout.
        margin_inner = 0.10
        margin_outer = safety_margin + 0.15              # loose outer (~0.45 m)
        mid = 0.5 * (margin_inner + margin_outer)

        m_in_eff  = mid + (margin_inner - mid) * turn
        m_out_eff = mid + (margin_outer - mid) * turn

        # Late-apex bias.
        # ----------------------------------------------
        # Classic racer trick: delay the turn-in so the corner can be
        # exited on a straighter line, carrying more speed onto the next
        # straight. Min-curvature alone gives the GEOMETRIC apex (middle
        # of corner); we want the TIME-optimal apex (closer to exit).
        #
        # We bias the asymmetric margin schedule by phase within a corner:
        #   - ENTRY  (|κ| INCREASING, dκ > 0):
        #       LOOSEN inner margin → line stays OUTER → delays turn-in
        #   - EXIT   (|κ| DECREASING, dκ < 0):
        #       TIGHTEN inner margin → line dives INNER → kills centrifugal
        #       spillout and aligns exit with the upcoming straight
        # The combined effect SHIFTS the apex location later in the corner,
        # which is exactly the late-apex / "drive deeper, exit straighter"
        # technique real racers use.
        dk = (np.roll(kmag_s, -1) - np.roll(kmag_s, 1)) * 0.5
        dk_norm = dk / max(np.abs(dk).max(), 1e-6)

        # Only consider these biases inside actual corners (turn > 0.2).
        in_corner = (turn > 0.2).astype(float)

        entry_grad = np.maximum( dk_norm, 0.0) * in_corner   # κ increasing
        exit_grad  = np.maximum(-dk_norm, 0.0) * in_corner   # κ decreasing

        # Smooth both biases (sigma=3) so the schedule is gradual.
        ebw = np.exp(-np.arange(-9, 10) ** 2 / (2 * 3.0 ** 2)); ebw /= ebw.sum()
        entry_bias_s = np.zeros_like(entry_grad)
        exit_bias_s  = np.zeros_like(exit_grad)
        for j in range(-9, 10):
            entry_bias_s += ebw[j + 9] * np.roll(entry_grad, -j)
            exit_bias_s  += ebw[j + 9] * np.roll(exit_grad,  -j)

        # Inner margin: loosened during entry, tightened during exit.
        entry_extra = 0.05      # [m] extra outer push at corner entry
        exit_extra  = 0.06      # [m] extra inner pull at corner exit
        m_in_eff = m_in_eff + entry_extra * entry_bias_s \
                            - exit_extra  * exit_bias_s
        m_in_eff = np.clip(m_in_eff, 0.05, margin_outer)

        # Two-phase budget (tuned via sweep).
        #
        # The shortest-path bias is used briefly (3 racing iters) to nudge
        # the line into the apex of every corner. After that, we run many
        # pure-curvature² iterations (15 polishing iters, ALL lambdas = 0)
        # which finalize the line as a clean min-curvature solution.
        #
        # Why this beats running pure min-curvature alone:
        #   Pure kappa² minimization has multiple local minima (especially
        #   on S-curves). The brief shortest-path bias acts as a "warm
        #   start" that steers the solver toward the apex-hugging local
        #   minimum — the racing line — rather than a centered solution.
        N_ITERS_RACE   = 3
        N_ITERS_POLISH = 15
        POLISH_S_MULT  = 0.0      # pure curvature² minimization in polish
        POLISH_J_MULT  = 0.0
        POLISH_L_MULT  = 0.0

        # Asymmetric alpha bounds along the ORIGINAL centerline normal.
        # κ_c > 0 (left turn): inner = LEFT (positive alpha) → use
        #                       tight margin on LEFT wall (hi)
        #                       loose margin on RIGHT wall (lo)
        # κ_c < 0 (right turn): inner = RIGHT (negative alpha) → opposite.
        # In our convention, hi = +w_l - margin_LEFT, lo = -w_r + margin_RIGHT.
        m_left  = np.where(ksign > 0, m_in_eff,  m_out_eff)
        m_right = np.where(ksign > 0, m_out_eff, m_in_eff)
        hi =  w_l_r - m_left
        lo = -w_r_r + m_right

        # --- Step 3 + 4. iterated min-curvature QP -------------------------
        # First pass linearizes about the centerline; later passes re-linearize
        # about the current optimized line, which sharpens the racing line in
        # corners where the lateral offset is large.
        xs, ys  = x_r.copy(), y_r.copy()
        a_total = np.zeros(N)

        # Identity for box-constraint matrix (reused every osqp solve).
        I_sparse = sparse.eye(N, format='csc')

        race_converged = False
        for it in range(N_ITERS_RACE + N_ITERS_POLISH):
            in_polishing = (it >= N_ITERS_RACE) or race_converged
            if in_polishing:
                lam_s = lambda_smooth * POLISH_S_MULT
                lam_j = lambda_jerk   * POLISH_J_MULT
                lam_l = lambda_short  * POLISH_L_MULT
            else:
                lam_s = lambda_smooth
                lam_j = lambda_jerk
                lam_l = lambda_short

            nx_, ny_ = _normals(xs, ys)

            # ---- Curvature operator (kappa ≈ 2nd diff of r along the loop) ----
            Ax = np.zeros((N, N))
            Ay = np.zeros((N, N))
            for i in range(N):
                im1, ip1 = (i - 1) % N, (i + 1) % N
                Ax[i, im1] =  nx_[im1]
                Ax[i, i  ] = -2.0 * nx_[i]
                Ax[i, ip1] =  nx_[ip1]
                Ay[i, im1] =  ny_[im1]
                Ay[i, i  ] = -2.0 * ny_[i]
                Ay[i, ip1] =  ny_[ip1]
            bx = np.roll(xs, -1) - 2.0 * xs + np.roll(xs, 1)
            by = np.roll(ys, -1) - 2.0 * ys + np.roll(ys, 1)

            # ---- Length operator (delta = forward 1st diff of r) -------------
            # Δr_i = (current_{i+1} - current_i) + (n_{i+1} da_{i+1} - n_i da_i)
            # Lx[i, i] = -nx_i ; Lx[i, i+1] = nx_{i+1}  (wrap-around)
            Lx = np.zeros((N, N))
            Ly = np.zeros((N, N))
            for i in range(N):
                ip1 = (i + 1) % N
                Lx[i, i  ] = -nx_[i]
                Lx[i, ip1] =  nx_[ip1]
                Ly[i, i  ] = -ny_[i]
                Ly[i, ip1] =  ny_[ip1]
            dx0 = np.roll(xs, -1) - xs   # baseline segment Δx with da = 0
            dy0 = np.roll(ys, -1) - ys

            # ---- Assemble the quadratic cost ---------------------------------
            #   min  daᵀH da + 2 gᵀ da
            #   s.t. lo_step ≤ da ≤ hi_step
            # The asymmetric bounds (lo, hi above) do most of the work for
            # apex hugging; the cost stays clean min-curvature + smoothing.
            H = (Ax.T @ Ax + Ay.T @ Ay              # curvature²
                 + lam_s * StS                       # alpha smoothness
                 + lam_j * TtT                       # curvature jerk
                 + lam_l * (Lx.T @ Lx + Ly.T @ Ly))  # length² (shortest path)
            g = (Ax.T @ bx + Ay.T @ by
                 + lam_l * (Lx.T @ dx0 + Ly.T @ dy0))

            lo_step = lo - a_total
            hi_step = hi - a_total

            # ---- Solve with osqp (proper QP solver) --------------------------
            # osqp form: min 0.5 xᵀPx + qᵀx  s.t. l ≤ Ax ≤ u
            # Our form:  min daᵀH da + 2 gᵀ da  →  P = 2H, q = 2g, A = I
            P = sparse.csc_matrix(2.0 * H)
            q = 2.0 * g
            prob = osqp.OSQP()
            prob.setup(
                P=P, q=q, A=I_sparse,
                l=lo_step, u=hi_step,
                verbose=False,
                max_iter=20000,
                eps_abs=1e-7, eps_rel=1e-7,
                polishing=True,            # extra clean-up on the osqp solution
            )
            res = prob.solve()
            if res.info.status_val not in (1, 2):
                # 1=solved, 2=solved_inaccurate; anything else is a failure mode.
                # Fall back to a feasible point (no change this iter) to avoid
                # crashing the whole optimization on a degenerate corner.
                da = np.zeros(N)
            else:
                da = np.asarray(res.x, dtype=float)
                # safety: clip strictly to the box (osqp may overshoot slightly)
                da = np.clip(da, lo_step, hi_step)

            xs = xs + da * nx_
            ys = ys + da * ny_
            a_total = a_total + da

            # If a racing pass converges, flip into polishing for the rest.
            if (not in_polishing) and np.max(np.abs(da)) < 5e-4:
                race_converged = True

            # If a racing pass converges, flip into the polishing phase for
            # the remaining iterations. Polishing passes never break early —
            # they must run to actually flatten the curvature spikes.
            if (not in_polishing) and np.max(np.abs(da)) < 5e-4:
                race_converged = True

        # --- Step 5. recover optimal raceline coordinates ------------------
        x_opt, y_opt = xs, ys
        a = a_total

        # --- Step 7. heading & curvature -----------------------------------
        psi, kappa = TrajectoryOptimizer._geom(x_opt, y_opt)

        # --- Step 8. speed profile (smooth braking + post-corner hold) -----
        # Pipeline:
        #   (1) cornering cap = sqrt(a_lat/kappa), capped by v_top
        #   (2) backward pass  → pre-corner braking. a_brake is REDUCED below
        #       a_long_max so deceleration starts EARLIER and ramps in
        #       gradually instead of snapping the speed down right at the
        #       corner entry.
        #   (3) POST-CORNER HOLD: keep the cornering speed for ~hold_dist
        #       after each corner.
        #   (4) forward pass   → gentle acceleration after the hold.
        #   (5) Gaussian smoothing of the final vx profile — kills any
        #       remaining stair-steps so the throttle/brake commands flow
        #       smoothly through the corner.
        v_top      = v_max
        a_accel    = 0.40 * a_long_max          # GENTLER exit accel → less outward "fling"
        a_brake    = 0.40 * a_long_max          # SOFTER, EARLIER braking
        hold_dist  = 6.0                        # [m] longer post-corner speed hold
                                                #     so the car doesn't ramp up too soon
                                                #     and get thrown into the outer wall

        N_pts = len(x_opt)
        ds = np.hypot(np.roll(x_opt, -1) - x_opt, np.roll(y_opt, -1) - y_opt)
        ds[ds < 1e-6] = 1e-6
        # Corner-cap bonus: bump corner cap speed up so entries don't feel
        # like a brick wall — pure-physics cornering limit is conservative,
        # the racing line has more grip available because the curvature is
        # spread evenly. Raised from 1.12 to 1.18 so the entry phase loses
        # less speed.
        vx_corner = np.sqrt(a_lat_max / np.maximum(np.abs(kappa), 1e-6))
        vx_cap = vx_corner * 1.18

        # ---- Lookahead-based v_top boost on long clear straights ---------
        # For each point i, walk forward along the raceline accumulating ds
        # until we hit a point whose |kappa| is above the curve threshold,
        # or we accumulate `lookahead_max` metres. The longer the clear
        # straight ahead, the higher we let v_top go locally.
        kappa_curve_thresh = 0.10       # [1/m] |κ| above this counts as curving
        lookahead_max      = 10.0       # [m]   look this far ahead to decide
        boost_max          = 1.45       # peak boost on a fully clear lookahead
        # On a short ~1 m clear stretch boost ≈ 1.04 (no speeding up).
        # On a full ~10 m clear stretch boost ≈ 1.45 (about 30% faster).
        kappa_abs = np.abs(kappa)
        straight_dist = np.zeros(N_pts)
        for i in range(N_pts):
            d_acc = 0.0
            j = i
            for _ in range(N_pts):
                if kappa_abs[j] > kappa_curve_thresh:
                    break
                d_acc += ds[j]
                if d_acc >= lookahead_max:
                    d_acc = lookahead_max
                    break
                j = (j + 1) % N_pts
            straight_dist[i] = d_acc
        boost = 1.0 + (boost_max - 1.0) * (straight_dist / lookahead_max)
        v_top_local = v_top * boost

        # Initialize vx as the min of (boosted v_top) and (corner cap)
        vx = np.minimum(v_top_local, vx_cap)

        # (2) Backward pass: pre-corner braking. Fewer passes than before so
        # the deceleration stays in a tighter window before the apex — the
        # car holds straight-line speed longer, then brakes a bit more
        # firmly. Combined with the cap bonus the result is "higher entry
        # speed, smooth braking, fast apex".
        for _ in range(7):
            for i in range(N_pts):
                j = (i - 1) % N_pts
                v_cap = np.sqrt(vx[i] ** 2 + 2.0 * a_brake * ds[j])
                vx[j] = min(vx[j], v_cap)

        # (3) Post-corner hold: each point inherits the minimum of the
        # previous hold_pts caps → corner speed gets extended forward,
        # blocking the forward pass from ramping up immediately.
        hold_pts = max(1, int(round(hold_dist / target_ds)))
        vx_held = vx.copy()
        for k in range(1, hold_pts + 1):
            vx_held = np.minimum(vx_held, np.roll(vx, k))
        vx = vx_held

        # (4) Forward pass: GENTLE acceleration limit (now 0.4 * a_long_max).
        # The lower a_accel + longer hold together kill the post-corner
        # outward fling: the car never ramps up speed until well after the
        # exit, so inertia stays manageable through corner exit.
        for _ in range(3):
            for i in range(N_pts):
                j = (i + 1) % N_pts
                v_cap = np.sqrt(vx[i] ** 2 + 2.0 * a_accel * ds[i])
                vx[j] = min(vx[j], v_cap)

        # (5) Gaussian smoothing on the closed-loop vx profile. Larger sigma
        # than before → speed changes spread over a longer arc, so corner
        # entry / exit feel gradual rather than stepped.
        vx_min_floor = vx.min()
        sigma_v = 2.5
        kk = max(1, int(round(3.0 * sigma_v)))
        wgt = np.exp(-np.arange(-kk, kk + 1) ** 2 / (2.0 * sigma_v ** 2))
        wgt /= wgt.sum()
        vx_s = np.zeros_like(vx)
        for jj in range(-kk, kk + 1):
            vx_s += wgt[jj + kk] * np.roll(vx, -jj)
        # Smoothing may only LOWER the profile (never above physical caps).
        vx = np.minimum(vx, vx_s)
        vx = np.maximum(vx, vx_min_floor * 0.95)

        # --- Step 9. remaining wall clearance ------------------------------
        w_r_new = w_r_r + a
        w_l_new = w_l_r - a

        return x_opt, y_opt, psi, kappa, vx, w_r_new, w_l_new

    # ======================================================================
    #                            HELPERS
    # ======================================================================
    @staticmethod
    def _load_centerline(path):
        """centerline.csv → (x, y, w_tr_right, w_tr_left) numpy arrays."""
        xs, ys, wrs, wls = [], [], [], []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(float(row['x_m']))
                ys.append(float(row['y_m']))
                wrs.append(float(row['w_tr_right_m']))
                wls.append(float(row['w_tr_left_m']))
        x = np.asarray(xs); y = np.asarray(ys)
        wr = np.asarray(wrs); wl = np.asarray(wls)
        # drop duplicate closing point if present
        if len(x) > 1 and np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-3:
            x, y, wr, wl = x[:-1], y[:-1], wr[:-1], wl[:-1]
        return x, y, wr, wl

    @staticmethod
    def _resample_uniform(x, y, w_r, w_l, target_ds):
        """Linear-interp resample of a closed loop onto uniform arc-length spacing."""
        seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
        s = np.concatenate(([0.0], np.cumsum(seg)))
        L = s[-1]
        N_new = max(20, int(round(L / target_ds)))
        s_new = np.linspace(0.0, L, N_new, endpoint=False)
        x_p  = np.concatenate((x,   [x[0]]))
        y_p  = np.concatenate((y,   [y[0]]))
        wr_p = np.concatenate((w_r, [w_r[0]]))
        wl_p = np.concatenate((w_l, [w_l[0]]))
        return (np.interp(s_new, s, x_p),
                np.interp(s_new, s, y_p),
                np.interp(s_new, s, wr_p),
                np.interp(s_new, s, wl_p))

    @staticmethod
    def _geom(x, y):
        """Heading psi and signed curvature kappa via centered differences (closed loop)."""
        dx  = (np.roll(x, -1) - np.roll(x, 1)) * 0.5
        dy  = (np.roll(y, -1) - np.roll(y, 1)) * 0.5
        ddx = np.roll(x, -1) - 2.0 * x + np.roll(x, 1)
        ddy = np.roll(y, -1) - 2.0 * y + np.roll(y, 1)
        psi = np.arctan2(dy, dx)
        denom = (dx * dx + dy * dy) ** 1.5
        denom[denom < 1e-9] = 1e-9
        kappa = (dx * ddy - dy * ddx) / denom
        return psi, kappa

    @staticmethod
    def _speed_profile(x, y, kappa, v_max, a_lat_max, a_long_max):
        """Point-mass speed profile: cornering limit + fwd/bwd accel smoothing."""
        N = len(x)
        ds = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
        ds[ds < 1e-6] = 1e-6
        v = np.minimum(v_max, np.sqrt(a_lat_max / np.maximum(np.abs(kappa), 1e-6)))
        # backward pass: braking limit
        for _ in range(2):
            for i in range(N):
                j = (i - 1) % N
                v_cap = np.sqrt(v[i] ** 2 + 2.0 * a_long_max * ds[j])
                v[j] = min(v[j], v_cap)
        # forward pass: acceleration limit
        for _ in range(2):
            for i in range(N):
                j = (i + 1) % N
                v_cap = np.sqrt(v[i] ** 2 + 2.0 * a_long_max * ds[i])
                v[j] = min(v[j], v_cap)
        return v

    @staticmethod
    def _save_global_waypoints(path, x, y, w_r, w_l, psi, kappa, vx):
        header = ['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m',
                  'psi_rad', 'kappa_radpm', 'vx_mps']
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(len(x)):
                w.writerow([f'{x[i]:.6f}', f'{y[i]:.6f}',
                            f'{w_r[i]:.4f}', f'{w_l[i]:.4f}',
                            f'{psi[i]:.6f}', f'{kappa[i]:.6f}',
                            f'{vx[i]:.4f}'])


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryOptimizer()
    rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
