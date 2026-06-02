"""Kinematic single-track bicycle model in JAX.

Adapted from opp_aware_mppi_f1tenth/mppi_example/dynamics_models/dynamics_models_jax.py
(vehicle_dynamics_ks), trimmed to F1TENTH params and a fixed-step RK4 integrator.

State x: [px, py, delta, v, psi]
Input u: [delta_dot, accel]  (already de-normalized to physical units)
"""

import jax
import jax.numpy as jnp

# F1TENTH params (from opp_aware_mppi_f1tenth mb_model_params.params_f1tenth)
LF = 0.15875
LR = 0.17145
LWB = LF + LR
S_MIN = -0.4189
S_MAX = 0.4189
V_MIN = -5.0
V_MAX = 20.0
SV_MIN = -3.2
SV_MAX = 3.2
V_SWITCH = 7.319
A_MAX = 9.51

# Defaults exposed to MPPI (control half-range for normalization).
DEFAULT_STEER_VEL_SCALE = SV_MAX  # rad/s
DEFAULT_ACCEL_SCALE = A_MAX       # m/s^2


def _steering_constraint(delta, sv):
    sv = jnp.where((delta <= S_MIN) & (sv <= 0.0), 0.0, sv)
    sv = jnp.where((delta >= S_MAX) & (sv >= 0.0), 0.0, sv)
    sv = jnp.clip(sv, SV_MIN, SV_MAX)
    return sv


def _accel_constraint(v, accl):
    pos_limit = jnp.where(v > V_SWITCH, A_MAX * V_SWITCH / jnp.maximum(v, 1e-3), A_MAX)
    accl = jnp.where((v <= V_MIN) & (accl <= 0.0), 0.0, accl)
    accl = jnp.where((v >= V_MAX) & (accl >= 0.0), 0.0, accl)
    accl = jnp.clip(accl, -A_MAX, pos_limit)
    return accl


def _f_ks(x, u):
    """Kinematic single-track ODE rhs. x: (5,), u: (2,)."""
    sv = _steering_constraint(x[2], u[0])
    a = _accel_constraint(x[3], u[1])
    return jnp.array([
        x[3] * jnp.cos(x[4]),
        x[3] * jnp.sin(x[4]),
        sv,
        a,
        x[3] / LWB * jnp.tan(x[2]),
    ])


def make_step(sim_dt, sub_dt=0.05):
    """Return a JIT-compiled one-step integrator with RK4 sub-stepping."""
    n_sub = max(1, int(round(sim_dt / sub_dt)))
    ddt = sim_dt / n_sub

    @jax.jit
    def step(x, u):
        def body(_, x0):
            k1 = _f_ks(x0, u)
            k2 = _f_ks(x0 + 0.5 * ddt * k1, u)
            k3 = _f_ks(x0 + 0.5 * ddt * k2, u)
            k4 = _f_ks(x0 + ddt * k3, u)
            return x0 + (ddt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return jax.lax.fori_loop(0, n_sub, body, x)

    return step
