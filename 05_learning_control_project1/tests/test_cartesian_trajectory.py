import numpy as np

from models.kinematics import TwoLinkKinematics

from evaluation.cartesian_trajectory import (
    get_stage_parameters,
    cartesian_reference,
    joint_reference,
)


def test_full_trajectory_reachability_and_continuity():
    kin = TwoLinkKinematics()

    for stage in (1, 2, 3):

        params = get_stage_parameters(stage)

        period = (
            2.0
            * np.pi
            / params.omega
        )

        times = np.linspace(
            0.0,
            period,
            1001,
        )

        q_history = []
        qd_history = []
        qdd_history = []

        min_abs_det_j = np.inf

        for t in times:

            p_d, _, _ = cartesian_reference(
                t=t,
                stage=stage,
            )

            q_d, qd_d, qdd_d = joint_reference(
                t=t,
                stage=stage,
                kinematics=kin,
                elbow="up",
            )

            # ----------------------------------------------
            # FK must reconstruct desired Cartesian position
            # ----------------------------------------------

            p_from_q = kin.forward_kinematics(
                q_d
            )

            assert np.allclose(
                p_from_q,
                p_d,
                atol=1e-10,
            )

            q_history.append(q_d)
            qd_history.append(qd_d)
            qdd_history.append(qdd_d)

            det_j = kin.jacobian_determinant(
                q_d
            )

            min_abs_det_j = min(
                min_abs_det_j,
                abs(det_j),
            )

        q_history = np.asarray(
            q_history
        )

        qd_history = np.asarray(
            qd_history
        )

        qdd_history = np.asarray(
            qdd_history
        )

        # ----------------------------------------------
        # Joint reference continuity
        # ----------------------------------------------

        delta_q = np.diff(
            q_history,
            axis=0,
        )

        max_joint_jump = np.max(
            np.abs(delta_q)
        )

        assert max_joint_jump < 0.05

        # ----------------------------------------------
        # Singularity margin
        # ----------------------------------------------

        assert min_abs_det_j > 1e-3

        # ----------------------------------------------
        # Numerical sanity
        # ----------------------------------------------

        assert np.all(
            np.isfinite(q_history)
        )

        assert np.all(
            np.isfinite(qd_history)
        )

        assert np.all(
            np.isfinite(qdd_history)
        )


def test_reference_closes_after_one_period():
    kin = TwoLinkKinematics()

    for stage in (1, 2, 3):

        params = get_stage_parameters(stage)

        period = (
            2.0
            * np.pi
            / params.omega
        )

        p_start, _, _ = cartesian_reference(
            t=0.0,
            stage=stage,
        )

        p_end, _, _ = cartesian_reference(
            t=period,
            stage=stage,
        )

        q_start, _, _ = joint_reference(
            t=0.0,
            stage=stage,
            kinematics=kin,
        )

        q_end, _, _ = joint_reference(
            t=period,
            stage=stage,
            kinematics=kin,
        )

        assert np.allclose(
            p_start,
            p_end,
            atol=1e-10,
        )

        assert np.allclose(
            q_start,
            q_end,
            atol=1e-10,
        )