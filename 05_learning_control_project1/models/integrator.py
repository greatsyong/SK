import numpy as np


def rk4_step(dynamics_func, state, dt, *args, **kwargs):
    """
    One RK4 integration step.

    Parameters
    ----------
    dynamics_func : callable
        Function returning state derivative.
    state : np.ndarray
        Current state.
    dt : float
        Integration timestep [s].
    *args, **kwargs :
        Additional arguments passed to dynamics_func.

    Returns
    -------
    np.ndarray
        State at next timestep.
    """

    k1 = dynamics_func(state, *args, **kwargs)

    k2 = dynamics_func(
        state + 0.5 * dt * k1,
        *args,
        **kwargs,
    )

    k3 = dynamics_func(
        state + 0.5 * dt * k2,
        *args,
        **kwargs,
    )

    k4 = dynamics_func(
        state + dt * k3,
        *args,
        **kwargs,
    )

    return state + (dt / 6.0) * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )