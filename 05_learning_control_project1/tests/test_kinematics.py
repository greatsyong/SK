import numpy as np
import pytest

from models.kinematics import TwoLinkKinematics


def test_fk_ik_fk_consistency():
    kin = TwoLinkKinematics()

    q_original = np.array(
        [-0.35, 1.45],
        dtype=np.float64,
    )

    position = kin.forward_kinematics(
        q_original
    )

    q_recovered = kin.inverse_kinematics(
        x=position[0],
        y=position[1],
        elbow="up",
    )

    position_recovered = kin.forward_kinematics(
        q_recovered
    )

    assert np.allclose(
        position,
        position_recovered,
        atol=1e-10,
    )


def test_both_ik_branches_reach_same_position():
    kin = TwoLinkKinematics()

    x = 0.45
    y = 0.15

    q_up = kin.inverse_kinematics(
        x=x,
        y=y,
        elbow="up",
    )

    q_down = kin.inverse_kinematics(
        x=x,
        y=y,
        elbow="down",
    )

    position_up = kin.forward_kinematics(
        q_up
    )

    position_down = kin.forward_kinematics(
        q_down
    )

    target = np.array(
        [x, y],
        dtype=np.float64,
    )

    assert np.allclose(
        position_up,
        target,
        atol=1e-10,
    )

    assert np.allclose(
        position_down,
        target,
        atol=1e-10,
    )

    assert q_up[1] > 0.0
    assert q_down[1] < 0.0


def test_unreachable_point_raises_error():
    kin = TwoLinkKinematics()

    with pytest.raises(ValueError):
        kin.inverse_kinematics(
            x=1.0,
            y=0.0,
            elbow="up",
        )


def test_jacobian_matches_finite_difference():
    kin = TwoLinkKinematics()

    q = np.array(
        [-0.30, 1.20],
        dtype=np.float64,
    )

    J_analytical = kin.jacobian(q)

    epsilon = 1e-7

    J_numerical = np.zeros(
        (2, 2),
        dtype=np.float64,
    )

    for i in range(2):
        dq = np.zeros(
            2,
            dtype=np.float64,
        )

        dq[i] = epsilon

        p_plus = kin.forward_kinematics(
            q + dq
        )

        p_minus = kin.forward_kinematics(
            q - dq
        )

        J_numerical[:, i] = (
            p_plus - p_minus
        ) / (
            2.0 * epsilon
        )

    assert np.allclose(
        J_analytical,
        J_numerical,
        atol=1e-7,
    )


def test_singularity_determinant():
    kin = TwoLinkKinematics()

    q_extended = np.array(
        [0.4, 0.0],
        dtype=np.float64,
    )

    determinant = kin.jacobian_determinant(
        q_extended
    )

    assert np.isclose(
        determinant,
        0.0,
        atol=1e-12,
    )