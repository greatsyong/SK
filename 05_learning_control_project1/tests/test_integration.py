import numpy as np

from models.parameters import TwoLinkParams
from models.two_link_dynamics import TwoLinkDynamics
from models.integrator import rk4_step


def simulate(
    robot,
    state0,
    tau,
    dt,
    duration,
):
    state = state0.copy()

    states = [state.copy()]
    times = [0.0]

    steps = int(duration / dt)

    for k in range(steps):
        state = rk4_step(
            robot.state_derivative,
            state,
            dt,
            tau,
        )

        states.append(state.copy())
        times.append((k + 1) * dt)

    return (
        np.array(times),
        np.array(states),
    )


def test_rk4_zero_state_zero_gravity():
    """
    If:
        qd = 0
        tau = 0
        gravity = 0
        friction = 0

    then the state must remain constant.
    """

    params = TwoLinkParams(
        g=0.0,
        b1=0.0,
        b2=0.0,
    )

    robot = TwoLinkDynamics(params)

    state0 = np.array([
        0.4,
        -0.3,
        0.0,
        0.0,
    ])

    tau = np.zeros(2)

    _, states = simulate(
        robot=robot,
        state0=state0,
        tau=tau,
        dt=0.001,
        duration=2.0,
    )

    assert np.allclose(
        states[-1],
        state0,
        atol=1e-10,
    )


def test_energy_conservation_no_gravity_no_friction():
    """
    Conservative system:
        gravity = 0
        friction = 0
        external torque = 0

    Kinetic energy should remain constant.
    """

    params = TwoLinkParams(
        g=0.0,
        b1=0.0,
        b2=0.0,
    )

    robot = TwoLinkDynamics(params)

    state = np.array([
        0.5,
        -0.7,
        1.0,
        -0.5,
    ])

    tau = np.zeros(2)

    dt = 0.001
    duration = 5.0
    steps = int(duration / dt)

    energies = []

    for _ in range(steps):
        q = state[:2]
        qd = state[2:]

        energies.append(
            robot.kinetic_energy(q, qd)
        )

        state = rk4_step(
            robot.state_derivative,
            state,
            dt,
            tau,
        )

    energies = np.array(energies)

    relative_drift = (
        np.max(np.abs(energies - energies[0]))
        / abs(energies[0])
    )

    assert relative_drift < 1e-6


def test_total_energy_conservation_with_gravity():
    """
    Conservative system:
        gravity ON
        friction OFF
        external torque = 0

    Total mechanical energy:
        E = T + V

    should remain approximately constant.
    """

    params = TwoLinkParams(
        b1=0.0,
        b2=0.0,
    )

    robot = TwoLinkDynamics(params)

    state = np.array([
        0.3,
        -0.5,
        0.4,
        -0.2,
    ])

    tau = np.zeros(2)

    dt = 0.001
    duration = 5.0
    steps = int(duration / dt)

    energies = []

    for _ in range(steps):
        q = state[:2]
        qd = state[2:]

        energies.append(
            robot.total_energy(q, qd)
        )

        state = rk4_step(
            robot.state_derivative,
            state,
            dt,
            tau,
        )

    energies = np.array(energies)

    scale = max(abs(energies[0]), 1.0)

    relative_drift = (
        np.max(np.abs(energies - energies[0]))
        / scale
    )

    assert relative_drift < 1e-5


def test_friction_dissipates_energy():
    """
    With viscous friction and no applied torque,
    mechanical energy should decrease overall.
    """

    params = TwoLinkParams()

    robot = TwoLinkDynamics(params)

    state = np.array([
        0.4,
        -0.3,
        0.8,
        -0.4,
    ])

    tau = np.zeros(2)

    dt = 0.001
    duration = 3.0
    steps = int(duration / dt)

    initial_energy = robot.total_energy(
        state[:2],
        state[2:],
    )

    for _ in range(steps):
        state = rk4_step(
            robot.state_derivative,
            state,
            dt,
            tau,
        )

    final_energy = robot.total_energy(
        state[:2],
        state[2:],
    )

    assert final_energy < initial_energy