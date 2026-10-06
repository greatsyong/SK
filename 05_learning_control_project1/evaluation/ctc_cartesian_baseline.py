import os
import numpy as np

from models.parameters import TwoLinkParams
from models.two_link_dynamics import TwoLinkDynamics
from models.integrator import rk4_step
from models.kinematics import TwoLinkKinematics

from controllers.computed_torque import ComputedTorqueController

from evaluation.cartesian_trajectory import (
    get_stage_parameters,
    cartesian_reference,
    joint_reference,
)


def run_stage(
    stage,
    control_dt=0.02,
    physics_dt=0.002,
):
    """
    Run one full Cartesian trajectory period using nominal CTC.

    The controller model and the physical plant use the same
    nominal parameters. This establishes the ideal model-based
    tracking baseline before introducing model mismatch or RL.
    """

    # ======================================================
    # Model / plant
    # ======================================================

    params = TwoLinkParams()

    plant = TwoLinkDynamics(
        params
    )

    kin = TwoLinkKinematics(
        params
    )

    controller = ComputedTorqueController(
        model=plant,
        kp=np.array(
            [100.0, 100.0]
        ),
        kd=np.array(
            [20.0, 20.0]
        ),
        torque_limits=None,
    )

    torque_limits = np.array(
        [
            params.tau1_max,
            params.tau2_max,
        ],
        dtype=np.float64,
    )

    # ======================================================
    # Stage duration
    # ======================================================

    traj_params = get_stage_parameters(
        stage
    )

    period = (
        2.0
        * np.pi
        / traj_params.omega
    )

    steps = int(
        round(
            period
            / control_dt
        )
    )

    physics_substeps = int(
        round(
            control_dt
            / physics_dt
        )
    )

    if not np.isclose(
        physics_substeps * physics_dt,
        control_dt,
    ):
        raise ValueError(
            "control_dt must be an integer multiple of physics_dt"
        )

    # ======================================================
    # Initial state
    # ======================================================

    q0, qd0, _ = joint_reference(
        t=0.0,
        stage=stage,
        kinematics=kin,
        elbow="up",
    )

    state = np.concatenate(
        [
            q0,
            qd0,
        ]
    ).astype(np.float64)

    # ======================================================
    # Logging
    # ======================================================

    ee_error_history = []
    joint_error_history = []
    torque_history = []

    ee_actual_history = []
    ee_reference_history = []

    # ======================================================
    # Simulation
    # ======================================================

    for k in range(steps):
        t = (
            k
            * control_dt
        )

        q = state[:2]
        qd = state[2:]

        q_ref, qd_ref, qdd_ref = joint_reference(
            t=t,
            stage=stage,
            kinematics=kin,
            elbow="up",
        )

        tau = controller.compute(
            q=q,
            qd=qd,
            q_ref=q_ref,
            qd_ref=qd_ref,
            qdd_ref=qdd_ref,
        )

        # Physical actuator limits
        tau = np.clip(
            tau,
            -torque_limits,
            torque_limits,
        )

        # Plant integration:
        # hold the control torque constant over the control interval
        # and integrate the plant with smaller physics substeps.
        for _ in range(physics_substeps):
            state = rk4_step(
                plant.state_derivative,
                state,
                physics_dt,
                tau,
            )

        # --------------------------------------------------
        # Evaluate against the reference at t + dt
        # --------------------------------------------------

        t_next = (
            t
            + control_dt
        )

        q_next = state[:2]

        p_actual = kin.forward_kinematics(
            q_next
        )

        p_ref, _, _ = cartesian_reference(
            t=t_next,
            stage=stage,
        )

        q_ref_next, _, _ = joint_reference(
            t=t_next,
            stage=stage,
            kinematics=kin,
            elbow="up",
        )

        ee_error = (
            p_ref
            - p_actual
        )

        joint_error = (
            q_ref_next
            - q_next
        )

        ee_error_history.append(
            ee_error
        )

        joint_error_history.append(
            joint_error
        )

        torque_history.append(
            tau
        )

        ee_actual_history.append(
            p_actual
        )

        ee_reference_history.append(
            p_ref
        )

    # ======================================================
    # Convert logs
    # ======================================================

    ee_error_history = np.asarray(
        ee_error_history
    )

    joint_error_history = np.asarray(
        joint_error_history
    )

    torque_history = np.asarray(
        torque_history
    )

    ee_actual_history = np.asarray(
        ee_actual_history
    )

    ee_reference_history = np.asarray(
        ee_reference_history
    )

    # ======================================================
    # Metrics
    # ======================================================

    ee_rmse_xy = np.sqrt(
        np.mean(
            ee_error_history**2,
            axis=0,
        )
    )

    ee_rmse_total = np.sqrt(
        np.mean(
            np.sum(
                ee_error_history**2,
                axis=1,
            )
        )
    )

    ee_max_error = np.max(
        np.linalg.norm(
            ee_error_history,
            axis=1,
        )
    )

    joint_rmse = np.sqrt(
        np.mean(
            joint_error_history**2,
            axis=0,
        )
    )

    torque_rms = np.sqrt(
        np.mean(
            torque_history**2,
            axis=0,
        )
    )

    torque_peak = np.max(
        np.abs(
            torque_history
        ),
        axis=0,
    )

    saturation_fraction = np.mean(
        np.isclose(
            np.abs(torque_history),
            torque_limits,
            rtol=0.0,
            atol=1e-6,
        ),
        axis=0,
    )

    return {
        "stage": stage,
        "period": period,

        "ee_rmse_x": ee_rmse_xy[0],
        "ee_rmse_y": ee_rmse_xy[1],
        "ee_rmse_total": ee_rmse_total,
        "ee_max_error": ee_max_error,

        "joint_rmse_q1": joint_rmse[0],
        "joint_rmse_q2": joint_rmse[1],

        "torque_rms_1": torque_rms[0],
        "torque_rms_2": torque_rms[1],

        "torque_peak_1": torque_peak[0],
        "torque_peak_2": torque_peak[1],

        "saturation_fraction_1": saturation_fraction[0],
        "saturation_fraction_2": saturation_fraction[1],

        "ee_actual": ee_actual_history,
        "ee_reference": ee_reference_history,
    }


def save_summary(results):
    """
    Save baseline metrics for later report generation.
    """

    os.makedirs(
        "results",
        exist_ok=True,
    )

    output_path = (
        "results/"
        "ctc_cartesian_baseline.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "CTC Cartesian Tracking Baseline\n"
        )

        f.write(
            "================================\n\n"
        )

        f.write(
            "Controller:\n"
        )

        f.write(
            "- Computed Torque Control\n"
        )

        f.write(
            "- Nominal model = true plant model\n"
        )

        f.write(
            "- Kp = [100, 100]\n"
        )

        f.write(
            "- Kd = [20, 20]\n"
        )

        f.write(
            "- Physical torque limits applied\n\n"
        )

        for result in results:

            f.write(
                f"Stage {result['stage']}\n"
            )

            f.write(
                "-------\n"
            )

            f.write(
                f"Period = "
                f"{result['period']:.6f} s\n"
            )

            f.write(
                f"EE RMSE x = "
                f"{result['ee_rmse_x']:.10f} m\n"
            )

            f.write(
                f"EE RMSE y = "
                f"{result['ee_rmse_y']:.10f} m\n"
            )

            f.write(
                f"EE RMSE total = "
                f"{result['ee_rmse_total']:.10f} m\n"
            )

            f.write(
                f"EE max error = "
                f"{result['ee_max_error']:.10f} m\n"
            )

            f.write(
                f"Joint RMSE q1 = "
                f"{result['joint_rmse_q1']:.10f} rad\n"
            )

            f.write(
                f"Joint RMSE q2 = "
                f"{result['joint_rmse_q2']:.10f} rad\n"
            )

            f.write(
                f"Torque RMS = "
                f"[{result['torque_rms_1']:.10f}, "
                f"{result['torque_rms_2']:.10f}] Nm\n"
            )

            f.write(
                f"Torque peak = "
                f"[{result['torque_peak_1']:.10f}, "
                f"{result['torque_peak_2']:.10f}] Nm\n"
            )

            f.write(
                f"Saturation fraction = "
                f"[{result['saturation_fraction_1']:.6f}, "
                f"{result['saturation_fraction_2']:.6f}]\n"
            )

            f.write("\n")

    return output_path


def main():
    results = []

    for stage in (
        1,
        2,
        3,
    ):
        result = run_stage(
            stage=stage,
        )

        results.append(
            result
        )

        print()
        print(
            "=" * 60
        )

        print(
            f"CTC BASELINE - STAGE {stage}"
        )

        print(
            "=" * 60
        )

        print(
            "Period [s]          :",
            f"{result['period']:.3f}",
        )

        print(
            "EE RMSE x [m]       :",
            f"{result['ee_rmse_x']:.10f}",
        )

        print(
            "EE RMSE y [m]       :",
            f"{result['ee_rmse_y']:.10f}",
        )

        print(
            "EE RMSE total [m]   :",
            f"{result['ee_rmse_total']:.10f}",
        )

        print(
            "EE max error [m]    :",
            f"{result['ee_max_error']:.10f}",
        )

        print(
            "Joint RMSE [rad]    :",
            [
                result["joint_rmse_q1"],
                result["joint_rmse_q2"],
            ],
        )

        print(
            "Torque RMS [Nm]     :",
            [
                result["torque_rms_1"],
                result["torque_rms_2"],
            ],
        )

        print(
            "Torque peak [Nm]    :",
            [
                result["torque_peak_1"],
                result["torque_peak_2"],
            ],
        )

        print(
            "Saturation fraction :",
            [
                result["saturation_fraction_1"],
                result["saturation_fraction_2"],
            ],
        )

    output_path = save_summary(
        results
    )

    print()
    print(
        "=" * 60
    )

    print(
        "Saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()