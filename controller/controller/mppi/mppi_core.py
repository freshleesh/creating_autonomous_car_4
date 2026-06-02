"""MPPI core loop in JAX.

Algorithm distilled from opp_aware_mppi_f1tenth/mppi_example/mppi_tracking.py:
sample truncated normal control perturbations around a warm-started mean,
rollout each sample through the kinematic ST dynamics, compute step rewards
(XY tracking + velocity + heading + opponent keep-out), turn into cumulative
return, softmax-weight by temperature, and update the mean. Cartesian-only.
"""

from functools import partial

import jax
import jax.numpy as jnp


class MPPI:
    """Fixed-shape MPPI tailored for waypoint tracking in cartesian frame.

    All shapes (n_samples, n_steps, MAX_OBS) are static at construction time so
    JAX compiles once. Runtime tunables (temperature, weights) pass through as
    traced arrays — no recompile.
    """

    A_SHAPE = 2  # [steering_velocity, accel]
    X_SHAPE = 5  # [px, py, delta, v, psi]

    def __init__(self, step_fn, n_samples, n_steps, max_obs,
                 control_std, norm_params, seed=0):
        self.step_fn = step_fn
        self.n_samples = int(n_samples)
        self.n_steps = int(n_steps)
        self.max_obs = int(max_obs)
        self.control_std = jnp.asarray(control_std, dtype=jnp.float32)  # (2,)
        self.norm_params = jnp.asarray(norm_params, dtype=jnp.float32)  # (2,) half-range
        self.key = jax.random.PRNGKey(int(seed))

        # Warm-start mean (in normalized units).
        self.a_opt = jnp.zeros((self.n_steps, self.A_SHAPE), dtype=jnp.float32)
        self.accum_matrix = jnp.triu(jnp.ones((self.n_steps, self.n_steps), dtype=jnp.float32))

    # ------------------------------------------------------------------
    # Rollout: one sample
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def _rollout(self, actions_norm, x0):
        # actions_norm: (n_steps, 2) in [-1, 1]
        actions = actions_norm * self.norm_params  # de-normalize

        def body(carry, u):
            x = self.step_fn(carry, u)
            return x, x

        _, states = jax.lax.scan(body, x0, actions)
        return states  # (n_steps, 5)

    # ------------------------------------------------------------------
    # Reward: per-step, summed inside `returns`
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def _step_reward(self, states, reference, obstacles, weights):
        """
        states     : (n_steps, 5)
        reference  : (n_steps, 4) — [ref_x, ref_y, ref_v, ref_psi]
        obstacles  : (max_obs, 2) — world XY; unused slots set far away
        weights    : (5,) — [w_xy, w_v, w_yaw, w_obs, obs_radius]
        """
        finite = jnp.isfinite(states).all(axis=1).astype(jnp.float32)
        invalid_pen = (1.0 - finite) * 1e3
        st = jnp.nan_to_num(states, nan=1e3, posinf=1e3, neginf=-1e3)

        dx = reference[:, 0] - st[:, 0]
        dy = reference[:, 1] - st[:, 1]
        xy_cost = jnp.sqrt(dx * dx + dy * dy)

        v_cost = jnp.abs(reference[:, 2] - st[:, 3])

        ref_psi = reference[:, 3]
        st_psi = st[:, 4]
        yaw_cost = (jnp.abs(jnp.sin(ref_psi) - jnp.sin(st_psi))
                    + jnp.abs(jnp.cos(ref_psi) - jnp.cos(st_psi)))

        # Opponent keep-out: soft Gaussian penalty per obstacle.
        # states_xy: (n_steps, 1, 2), obstacles: (1, max_obs, 2)
        d_obs = states[:, None, :2] - obstacles[None, :, :]      # (n_steps, max_obs, 2)
        d2 = jnp.sum(d_obs * d_obs, axis=-1)                     # (n_steps, max_obs)
        radius = jnp.maximum(weights[4], 1e-3)
        opp_cost = jnp.sum(jnp.exp(-d2 / (2.0 * radius * radius)), axis=-1)  # (n_steps,)

        reward = (
            -weights[0] * xy_cost
            - weights[1] * v_cost
            - weights[2] * yaw_cost
            - weights[3] * opp_cost
            - invalid_pen
        )
        return reward  # (n_steps,)

    # ------------------------------------------------------------------
    # Update: one MPPI iteration
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=0)
    def _iteration(self, a_opt, key, x0, reference, obstacles, weights,
                   temperature, damping):
        key, sub = jax.random.split(key)
        lo = -1.0 - a_opt
        hi = 1.0 - a_opt
        da = jax.random.truncated_normal(
            sub, lo / self.control_std, hi / self.control_std,
            shape=(self.n_samples, self.n_steps, self.A_SHAPE),
        ) * self.control_std

        actions = jnp.clip(a_opt[None] + da, -1.0, 1.0)         # (S, T, 2)

        states = jax.vmap(self._rollout, in_axes=(0, None))(actions, x0)  # (S, T, 5)
        rewards = jax.vmap(self._step_reward, in_axes=(0, None, None, None))(
            states, reference, obstacles, weights,
        )                                                        # (S, T)
        rewards = jnp.nan_to_num(rewards, nan=-1e6, posinf=-1e6, neginf=-1e6)

        # Cumulative return-to-go per step: R[s, t] = sum_{k>=t} r[s, k]
        # Using upper-triangular accumulator matches mppi_tracking.returns.
        returns = rewards @ self.accum_matrix.T                  # (S, T)
        returns = jnp.nan_to_num(returns, nan=-1e6, posinf=-1e6, neginf=-1e6)

        # Softmax weights per time step (column-wise normalization).
        r_max = jnp.max(returns, axis=0, keepdims=True)
        r_min = jnp.min(returns, axis=0, keepdims=True)
        denom = jnp.maximum((r_max - r_min) + damping, 1e-6)
        scaled = (returns - r_max) / denom                       # (S, T)
        w = jnp.exp(scaled / jnp.maximum(temperature, 1e-6))     # (S, T)
        w_sum = jnp.sum(w, axis=0, keepdims=True)
        uniform = jnp.ones_like(w) / w.shape[0]
        w = jnp.where(w_sum > 0.0, w / w_sum, uniform)           # (S, T)

        # da: (S, T, 2); w: (S, T) → weighted mean over samples per step+axis.
        da_opt = jnp.sum(da * w[:, :, None], axis=0)             # (T, 2)
        a_opt = jnp.clip(a_opt + da_opt, -1.0, 1.0)

        # Best-roll trajectory for visualization.
        traj_opt = self._rollout(a_opt, x0)
        return a_opt, key, traj_opt

    # ------------------------------------------------------------------
    # Public update
    # ------------------------------------------------------------------
    def update(self, x0, reference, obstacles, weights,
               temperature=0.01, damping=0.001, n_iter=1):
        x0 = jnp.asarray(x0, dtype=jnp.float32)
        reference = jnp.asarray(reference, dtype=jnp.float32)
        obstacles = jnp.asarray(obstacles, dtype=jnp.float32)
        weights = jnp.asarray(weights, dtype=jnp.float32)
        temperature = jnp.asarray(temperature, dtype=jnp.float32)
        damping = jnp.asarray(damping, dtype=jnp.float32)

        # Shift warm-start: drop first step, append zero.
        a_opt = jnp.concatenate(
            [self.a_opt[1:], jnp.zeros((1, self.A_SHAPE), dtype=jnp.float32)],
            axis=0,
        )

        traj_opt = None
        for _ in range(max(1, int(n_iter))):
            a_opt, self.key, traj_opt = self._iteration(
                a_opt, self.key, x0, reference, obstacles, weights,
                temperature, damping,
            )
        self.a_opt = a_opt
        return a_opt, traj_opt

    def reset_warm_start(self):
        self.a_opt = jnp.zeros((self.n_steps, self.A_SHAPE), dtype=jnp.float32)
