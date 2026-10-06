import numpy as np

from models.parameters import TwoLinkParams
from models.two_link_dynamics import TwoLinkDynamics


params = TwoLinkParams()
robot = TwoLinkDynamics(params)


def test_mass_matrix_symmetry():
    for _ in range(1000):
        q = np.random.uniform(-np.pi, np.pi, size=2)

        M = robot.mass_matrix(q)

        assert np.allclose(
            M,
            M.T,
            atol=1e-12,
        )


def test_mass_matrix_positive_definite():
    for _ in range(1000):
        q = np.random.uniform(-np.pi, np.pi, size=2)

        M = robot.mass_matrix(q)

        eigvals = np.linalg.eigvalsh(M)

        assert np.all(eigvals > 0.0)


def test_gravity_horizontal_configuration():
    q = np.array([0.0, 0.0])

    g = robot.gravity_vector(q)

    expected_g1 = (
        params.m1 * params.lc1
        + params.m2 * params.l1
        + params.m2 * params.lc2
    ) * params.g

    expected_g2 = (
        params.m2
        * params.lc2
        * params.g
    )

    assert np.allclose(
        g,
        np.array([expected_g1, expected_g2]),
        atol=1e-12,
    )


def test_static_gravity_compensation():
    q = np.array([0.3, -0.5])
    qd = np.zeros(2)

    tau = robot.gravity_vector(q)

    qdd = robot.forward_dynamics(
        q=q,
        qd=qd,
        tau=tau,
    )

    assert np.allclose(
        qdd,
        np.zeros(2),
        atol=1e-10,
    )


def test_forward_inverse_dynamics_consistency():
    for _ in range(1000):
        q = np.random.uniform(
            -np.pi,
            np.pi,
            size=2,
        )

        qd = np.random.uniform(
            -2.0,
            2.0,
            size=2,
        )

        qdd_ref = np.random.uniform(
            -5.0,
            5.0,
            size=2,
        )

        tau = robot.inverse_dynamics(
            q=q,
            qd=qd,
            qdd=qdd_ref,
        )

        qdd = robot.forward_dynamics(
            q=q,
            qd=qd,
            tau=tau,
        )

        assert np.allclose(
            qdd,
            qdd_ref,
            atol=1e-10,
        )