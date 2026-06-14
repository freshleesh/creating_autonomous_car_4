#!/usr/bin/env python3

import os
import csv
import numpy as np
import osqp
from scipy import sparse
from scipy.interpolate import CubicSpline

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
        self.declare_parameter('margin_inner',  0.30)   # [m]   margin to the inside-of-corner wall
        self.declare_parameter('margin_outer',  -1.0)   # [m]   margin to the outside-of-corner wall (<0 → safety_margin + 0.15)
        self.declare_parameter('v_max',         8.0)    # [m/s] vehicle speed cap
        self.declare_parameter('a_lat_max',     6.0)    # [m/s^2] lateral grip limit
        self.declare_parameter('a_long_max',    4.0)    # [m/s^2] longitudinal accel limit
        self.declare_parameter('target_ds',     0.25)   # [m]   uniform ds for QP input/output
        # [user] High-speed early-braking: where the planned straight speed is high,
        #   lower a_brake so deceleration into the next corner starts EARLIER.
        self.declare_parameter('brake_hi_factor', 0.80)  # a_brake multiplier at high speed (1.0=off, lower=brake earlier)
        self.declare_parameter('brake_hi_v_lo',   0.85)  # ×v_max: reduction starts ramping above this planned speed
        self.declare_parameter('brake_hi_v_hi',   1.10)  # ×v_max: full reduction at/above this planned speed

        map_name      = self.get_parameter('map_name').value
        input_csv     = self.get_parameter('input_csv').value
        output_csv    = self.get_parameter('output_csv').value
        safety_margin = self.get_parameter('safety_margin').value
        margin_inner  = self.get_parameter('margin_inner').value
        margin_outer  = self.get_parameter('margin_outer').value
        if margin_outer < 0.0:                       # backward-compat fallback
            margin_outer = safety_margin + 0.15
        v_max         = self.get_parameter('v_max').value
        a_lat_max     = self.get_parameter('a_lat_max').value
        a_long_max    = self.get_parameter('a_long_max').value
        target_ds     = self.get_parameter('target_ds').value
        brake_hi_factor = self.get_parameter('brake_hi_factor').value
        brake_hi_v_lo   = self.get_parameter('brake_hi_v_lo').value
        brake_hi_v_hi   = self.get_parameter('brake_hi_v_hi').value

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
            f'(margin={safety_margin}, m_in={margin_inner}, m_out={margin_outer}, '
            f'v_max={v_max}, a_lat={a_lat_max}, '
            f'a_long={a_long_max}, target_ds={target_ds})'
        )
        x_opt, y_opt, psi, kappa, vx, w_r_new, w_l_new = self._optimize(
            x_c, y_c, w_r, w_l,
            safety_margin=safety_margin,
            margin_inner=margin_inner,
            margin_outer=margin_outer,
            v_max=v_max,
            a_lat_max=a_lat_max,
            a_long_max=a_long_max,
            target_ds=target_ds,
            brake_hi_factor=brake_hi_factor,
            brake_hi_v_lo=brake_hi_v_lo,
            brake_hi_v_hi=brake_hi_v_hi,
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
                  safety_margin, margin_inner, margin_outer,
                  v_max, a_lat_max, a_long_max, target_ds,
                  brake_hi_factor=1.0, brake_hi_v_lo=0.85, brake_hi_v_hi=1.10):
        """
        Minimum-curvature trajectory optimization.

        Parameters
        ----------
        x_c, y_c   : (N,) centerline coordinates (closed loop, no duplicate end)
        w_r, w_l   : (N,) track half-widths to the right / left walls
        safety_margin : [m] base wall clearance (used as margin_outer fallback)
        margin_inner  : [m] margin to the inside-of-corner wall  (yaml: margin_inner)
        margin_outer  : [m] margin to the outside-of-corner wall (yaml: margin_outer)
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
        # Lowered to ~anti-jitter level only. The final Step-6 cubic-spline
        # resample now handles smoothing, so we no longer need a strong 2nd-
        # difference penalty here. A large lambda_smooth pulls the line toward
        # the centerline and prevents the apex dive — exactly the "corners too
        # wide" symptom — so we keep it just high enough to condition the QP.
        lambda_smooth = 0.02

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
        lambda_jerk = 1.0         # raised: smoother curvature transitions at corner entry/exit

        # Shortest-path term (Option D).
        # L² = Σ|Δr|², quadratic in da → adds to H and g as λ_short · L².
        # DISABLED: the shortest-path bias cuts corners (pulls the line to the
        # inside apex with a small radius), which RAISES peak curvature and
        # lowers corner speed. Pure curvature² minimization spreads the
        # curvature instead → lower peak κ, higher corner speed. This is the
        # core of phase A.
        lambda_short = 0.0

        # Speed-weighted curvature (min-time approximation). Applied during the
        # racing iterations only; polishing then runs pure curvature² to clean
        # up. See the detailed note where wc is built. Set USE_TIME_WEIGHT to
        # False to fall back to plain geometric min-curvature.
        USE_TIME_WEIGHT = True
        TIME_WEIGHT_CAP = 9.0     # cap on (v_max/v_local)² so one slow apex
                                  # cannot dominate the entire objective

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

        # ---- Long-straight curvature penalty (predict next corner) ---------
        # Min-curvature spreads a fixed lateral shift over the WHOLE straight
        # (a long gentle S) because spreading lowers Σκ². Visually that is the
        # "drift to center and back out" on straights. A racer instead drives
        # the straight as a STRAIGHT CHORD (zero curvature, max speed / shortest
        # distance) and compresses all the lateral repositioning into the
        # corner zones (out–in–out).
        #
        # To force that, we measure how far each point is from the nearest
        # corner (along the track) and multiply the curvature weight UP on long
        # straights. Curvature there becomes expensive, so the solver flattens
        # the straight and pushes the transition into the corner entry/exit,
        # where the weight drops back to ~1 and the line is free to move. The
        # corner mask comes from the centerline curvature, so this "looks
        # ahead" to the next corner purely from track geometry.
        is_corner = turn > 0.20
        ds_u = target_ds
        big  = N * ds_u
        dist_corner = np.full(N, big)
        d = big
        for i in range(2 * N):                 # forward sweep (wrap-around)
            idx = i % N
            d = 0.0 if is_corner[idx] else d + ds_u
            if d < dist_corner[idx]:
                dist_corner[idx] = d
        d = big
        for i in range(2 * N - 1, -1, -1):     # backward sweep (wrap-around)
            idx = i % N
            d = 0.0 if is_corner[idx] else d + ds_u
            if d < dist_corner[idx]:
                dist_corner[idx] = d
        STRAIGHT_REF   = 6.0    # [m] distance-from-corner where straightening saturates
        STRAIGHT_BOOST = 5.0    # max extra curvature weight on a long clear straight
        straight_w = 1.0 + STRAIGHT_BOOST * np.clip(dist_corner / STRAIGHT_REF, 0.0, 1.0)

        # margin_inner / margin_outer now come from trajectory_optimizer.yaml.
        # Typical racing setup: inner < outer (tight apex, loose outer wall).
        # inner > outer is also allowed (pushes the line OFF the inner wall,
        # useful when PP cuts inside) — the clip below is order-safe.
        mid = 0.5 * (margin_inner + margin_outer)

        m_in_eff  = mid + (margin_inner - mid) * turn
        m_out_eff = mid + (margin_outer - mid) * turn

        # Late-apex bias.cvzxcv
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

        # Inner margin: late-apex entry/exit bias DISABLED for phase A.
        # These hacks were pushing the line outward at corner entry (part of
        # the "corners too wide" symptom). We first want to see the clean,
        # pure min-curvature racing line; re-introduce a small bias later only
        # if needed.
        entry_extra = 0.0       # [m] extra outer push at corner entry (disabled)
        exit_extra  = 0.0       # [m] extra inner pull at corner exit (disabled)
        m_in_eff = m_in_eff + entry_extra * entry_bias_s \
                            - exit_extra  * exit_bias_s
        # Order-safe clip: np.clip(x, lo, hi) silently returns hi everywhere
        # when lo > hi, which used to erase any margin_inner > margin_outer
        # setting. Sort the bounds so both orderings work as intended.
        m_lo = min(margin_inner, margin_outer)
        m_hi = max(margin_inner, margin_outer)
        m_in_eff = np.clip(m_in_eff, m_lo, m_hi)

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

        # Pinch-point guard: at narrow sections the margins can exceed the
        # track width and invert the corridor (lo > hi), which makes OSQP
        # reject the problem ("lower bound > upper bound"). Collapse such
        # points to the band midpoint so we always pass l <= u.
        bad = lo > hi
        mid_band = 0.5 * (lo + hi)
        lo = np.where(bad, mid_band, lo)
        hi = np.where(bad, mid_band, hi)

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

            # ---- Arc-length curvature correction -----------------------------
            # The plain 2nd-difference above approximates d²r/di² (index
            # parametrization). The true curvature is d²r/ds² ≈ 2nddiff / ds²,
            # so on a UNIFORM grid 2nddiff ≈ ds²·κ and minimizing Σ(2nddiff)²
            # ≈ minimizing Σκ². But the optimized line bunches points on the
            # inside of corners (ds shrinks there); the uniform assumption
            # then UNDER-counts curvature exactly at the tightest apex, so the
            # QP leaves a sharp single-point spike (the "pointy hairpin").
            # Re-weighting each row by (target_ds/ds_i)² rescales every point
            # back to a common ds, so the objective is Σκ² regardless of local
            # spacing — the solver now flattens the apex spike instead of
            # ignoring it.
            hf = np.hypot(np.roll(xs, -1) - xs, np.roll(ys, -1) - ys)
            ds_loc = 0.5 * (hf + np.roll(hf, 1))
            ds_loc = np.maximum(ds_loc, 0.3 * target_ds)   # clamp → no weight blow-up
            wc = (target_ds / ds_loc) ** 2                 # ≈1 on uniform spacing

            # ---- Speed-weighted curvature (min-time approximation) -----------
            # Pure Σκ² minimization gives the GEOMETRIC apex: it refuses to add
            # the entry/exit "setup S" that racers use because that S costs
            # curvature. The result is a line that returns toward the
            # centerline on straights instead of staying wide to set up the
            # next corner (the "why does it pull to center then leave" you saw).
            #
            # Min-TIME wants the opposite: maximise the radius at the SLOW apex
            # (the lap-time-limiting point) even at the cost of more curvature
            # on the fast entry, because lap time ∝ ∫ds/v and v is grip-limited
            # at the apex. Weighting each curvature row by (v_max / v_local)²
            # does exactly this — slow apex points are penalised hard, so the
            # solver opens the corner by using the full entry/exit width
            # (out–in–out), while fast straights (weight≈1) stay free to carry
            # the setup curvature. This recovers the racing line without a full
            # dynamic min-time NLP.
            # Speed (time) weight is DYNAMIC — it recomputes v from the current
            # curvature every iteration. Run through the polishing phase it
            # feeds back on itself across the re-linearizations and diverges
            # (twisty maps blow up). So it stays in the racing phase only, as a
            # gentle apex-opening warm start.
            if USE_TIME_WEIGHT and not in_polishing:
                _, k_cur = TrajectoryOptimizer._geom(xs, ys)
                v_cur = np.sqrt(a_lat_max / np.maximum(np.abs(k_cur), 1e-6))
                v_cur = np.clip(v_cur, 0.2 * v_max, v_max)
                w_spd = np.minimum((v_max / v_cur) ** 2, TIME_WEIGHT_CAP)
                wc = wc * w_spd

            # Long-straight penalty is STATIC (fixed from the centerline
            # geometry), so it is stable to apply on EVERY iteration — polishing
            # included. This matters: if it were racing-only, the 15
            # pure-curvature² polish iters would re-spread curvature back onto
            # the straights and undo the straightening. Applied throughout, the
            # straights stay flat chords and the lateral transition lives in the
            # corner zones (out–in–out).
            wc = wc * straight_w

            Ax *= wc[:, None]; bx *= wc
            Ay *= wc[:, None]; by *= wc

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

        # --- Step 5 + 6. recover line, then dedupe + cubic-spline resample -
        # After the QP iterations, points are bunched on the inside of corners
        # (lateral moves compress the arc spacing there). The curvature
        # operator and _geom both assume UNIFORM ds, so on the clustered
        # points _geom reports fake curvature spikes — that is the "pointy
        # corner" symptom. We resample the optimized line onto uniform arc
        # length with a periodic cubic spline before computing psi/kappa.
        # The alpha offset and the (resampled) centerline half-widths are
        # carried along so the remaining wall clearance stays consistent.
        x_opt, y_opt, a, w_r_r, w_l_r = TrajectoryOptimizer._resample_closed(
            xs, ys, target_ds, a_total, w_r_r, w_l_r)

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
        a_accel    = 25.80 * a_long_max          # [revert v10→v8.1] gentler accel out of corners (0.85 carried too much speed)
        a_brake    = 4.50* a_long_max          # lowered 1.5→1.2: slightly longer/gentler braking zone (∝1/a_brake)
                                                #   but physical: braking-zone length = Δv²/(2·a_brake), so a
                                                #   LOWER a_brake stretches the slow-down EARLIER. The car
                                                #   "brakes too early" precisely because a_brake was low. A
                                                #   higher a_brake = short, late slow-down held near the corner.
                                                #   (Trade: the braking itself is firmer — opposite of "gentler".)
        hold_dist  = 0.1                       # [revert v10→v8.1] small post-corner hold restored:
                                                #     a touch of corner-speed hold past the apex = gentler
                                                #     exit, less over-speed into the next section.

        N_pts = len(x_opt)
        ds = np.hypot(np.roll(x_opt, -1) - x_opt, np.roll(y_opt, -1) - y_opt)
        ds[ds < 1e-6] = 1e-6
        # Corner-cap bonus, now CURVATURE-GRADED.
        # The pure point-mass limit sqrt(a_lat/κ) is conservative because the
        # racing line spreads curvature and has grip headroom. How much extra
        # we dare take depends on how hard the corner is:
        #   - gentle corners (small |κ|, large radius): plenty of margin → push
        #     the cap up a lot (cap_mild).
        #   - severe corners (large |κ|, tight radius): stay close to physics
        #     (cap_sharp) so we don't ask for grip the tyres don't have.
        # We blend linearly from cap_mild to cap_sharp as |κ| rises to
        # KAPPA_HARD, so weak corners speed up the most while hairpins stay
        # safe.
        # cap_factor NEUTRALIZED to 1.0. The cornering speed limit now lives here
        # in the trajectory (a_lat_max=15 from yaml) and must equal PP's old NEAR
        # hard cap exactly: vx_cap = √(a_lat_max/κ)·cap_factor → √(15/κ) when
        # cap_factor=1.0. The old 2.0–3.0 "grip-headroom bonus" is gone because PP
        # no longer re-caps corner speed; the trajectory profile IS the limit now.
        cap_mild   = 1.70       # gentle-corner grip-headroom bonus (×√(a_lat/κ)). raised 1.4→1.6
        cap_sharp  = 1.50       # sharp-corner bonus (smaller; tight corners stay closer to physics). 1.2→1.4
        KAPPA_HARD = 1.50       # [1/m] |κ| at/above which we treat a corner as "severe"
        t_sharp = np.clip(np.abs(kappa) / KAPPA_HARD, 0.0, 1.0)   # 0 mild .. 1 sharp
        cap_factor = cap_mild + (cap_sharp - cap_mild) * t_sharp
        vx_corner = np.sqrt(a_lat_max / np.maximum(np.abs(kappa), 1e-6))
        vx_cap = vx_corner * cap_factor

        # ---- Lookahead-based v_top boost on long clear straights ---------
        # For each point i, walk forward along the raceline accumulating ds
        # until we hit a point whose |kappa| is above the curve threshold,
        # or we accumulate `lookahead_max` metres. The longer the clear
        # straight ahead, the higher we let v_top go locally.
        kappa_curve_thresh = 0.10       # [1/m] |κ| above this counts as curving
        lookahead_max      = 10.0       # [m]   look this far ahead to decide
        boost_max          = 1.35        # straight over-boost OFF → straights = v_max exactly (raise >1 here, PP boost stays off, for long-straight boost).
        # The old 1.90 pushed the straight target to v_max·1.90 (=15.2 m/s at
        # v_max=8) which is the main reason straights ran "way too fast". With
        # boost_max=1.0 the straight target is exactly v_max — predictable and
        # tame. Re-raise (e.g. 1.2–1.5) later once the baseline feels stable.
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

        # [user] High-speed early-braking. On long straights the planned speed
        #   climbs high; with a CONSTANT a_brake the braking zone (length
        #   ∝ Δv²/a_brake) starts too late and the car can't shed the speed
        #   before the corner. Where the planned speed v_ref is high we REDUCE
        #   a_brake → longer braking zone → deceleration begins EARLIER. Linear
        #   ramp: no change below brake_hi_v_lo·v_max, full brake_hi_factor at/
        #   above brake_hi_v_hi·v_max. brake_hi_factor=1.0 disables this entirely.
        #   v_ref is the PRE-braking cap (straight/corner speed), so the per-point
        #   a_brake is fixed before propagation and stays stable across passes.
        #   NOTE: this only bites in actual braking zones — on an open straight the
        #   downstream point is also fast, so v_cap stays high and nothing slows.
        v_ref  = vx.copy()
        _v_lo  = brake_hi_v_lo * v_max
        _v_hi  = max(brake_hi_v_hi * v_max, _v_lo + 1e-6)
        _t_fast = np.clip((v_ref - _v_lo) / (_v_hi - _v_lo), 0.0, 1.0)
        a_brake_arr = a_brake * (1.0 - (1.0 - brake_hi_factor) * _t_fast)

        # (2) Backward pass: pre-corner braking. Fewer passes than before so
        # the deceleration stays in a tighter window before the apex — the
        # car holds straight-line speed longer, then brakes a bit more
        # firmly. Combined with the cap bonus the result is "higher entry
        # speed, smooth braking, fast apex".
        for _ in range(7):
            for i in range(N_pts):
                j = (i - 1) % N_pts
                v_cap = np.sqrt(vx[i] ** 2 + 2.0 * a_brake_arr[j] * ds[j])
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

        # (6) FINAL braking-limit pass. The post-corner hold (3) and the
        # Gaussian smoothing (5) both run AFTER the main backward pass (2), and
        # they can re-introduce local decelerations far steeper than a_brake at
        # corner-cap edges (measured spikes of >12 m/s² before this pass). Those
        # are exactly the "braking is too harsh" jolts. Re-apply the braking
        # limit one more time as the LAST step so the final profile decelerates
        # no harder than a_brake ANYWHERE → gentle, gradual slow-downs into
        # corners. (Backward pass only lowers speeds, so it never breaks the
        # corner caps; it just starts the slow-down a little earlier.)
        # Sweep DESCENDING so each point is capped against its ALREADY-finalized
        # successor: the braking limit then propagates all the way back through a
        # long approach in a single sweep. (An ascending sweep — as the earlier
        # passes use — only moves the limit back one point per iteration, so long
        # braking zones stay under-capped and keep steep >a_brake drops.)
        for _ in range(3):
            for j in range(N_pts - 1, -1, -1):
                i = (j + 1) % N_pts
                v_cap = np.sqrt(vx[i] ** 2 + 2.0 * a_brake_arr[j] * ds[j])
                if vx[j] > v_cap:
                    vx[j] = v_cap

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
    def _resample_closed(x, y, target_ds, *extra):
        """Periodic cubic-spline resample of a closed (x, y) loop onto uniform
        arc length. Any `extra` per-point arrays (alpha offset, half-widths,
        …) are linearly interpolated onto the same grid and returned after
        the new x, y. De-clusters points bunched on the inside of corners so
        the centered-difference curvature in _geom stays well conditioned and
        does not produce fake spikes."""
        seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
        s = np.concatenate(([0.0], np.cumsum(seg)))
        L = s[-1]
        N_new = max(20, int(round(L / target_ds)))
        s_new = np.linspace(0.0, L, N_new, endpoint=False)
        # periodic cubic spline needs matching endpoints (closed loop)
        xp = np.concatenate((x, [x[0]]))
        yp = np.concatenate((y, [y[0]]))
        csx = CubicSpline(s, xp, bc_type='periodic')
        csy = CubicSpline(s, yp, bc_type='periodic')
        out = [csx(s_new), csy(s_new)]
        for e in extra:
            ep = np.concatenate((np.asarray(e), [e[0]]))
            out.append(np.interp(s_new, s, ep))
        return out

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
