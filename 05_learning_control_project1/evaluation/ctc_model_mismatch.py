from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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


# ==========================================================
# Experiment configuration
# ==========================================================

STAGE = 3

CONTROL_DT = 0.02       # 50 Hz controller
PHYSICS_DT = 0.002      # 500 Hz physics
CYCLES = 3

MODEL_M2_SCALE = 0.70   # controller underestimates m2 by 30%


def main():

    # ======================================================
    # True physical plant
    # ======================================================

    true_params = TwoLinkParams()

    true_plant = TwoLinkDynamics(
        true_params
    )

    kinematics = TwoLinkKinematics(
        true_params
    )

    torque_limits = np.array(
        [
            true_params.tau1_max,
            true_params.tau2_max,
        ],
        dtype=np.float64,
    )

    # ======================================================
    # Imperfect model used only by CTC
    # ======================================================
    #
    # IMPORTANT:
    #
    # The physical plant remains unchanged.
    #
    # Only the controller's internal estimate of link-2 mass
    # is modified. This creates a controlled model mismatch
    # while keeping every other model parameter exact.
    #
    # true:
    #     m2 = 1.50 kg
    #
    # controller model:
    #     m2 = 1.05 kg
    #
    # ======================================================

    controller_params = replace(
        true_params,
        m2=(
            true_params.m2
            * MODEL_M2_SCALE
        ),
    )

    controller_model = TwoLinkDynamics(
        controller_params
    )

    controller = ComputedTorqueController(
        model=controller_model,
        kp=np.array(
            [100.0, 100.0],
            dtype=np.float64,
        ),
        kd=np.array(
            [20.0, 20.0],
            dtype=np.float64,
        ),
        torque_limits=None,
    )

    # ======================================================
    # Timing
    # ======================================================

    ratio = CONTROL_DT / PHYSICS_DT

    if not np.isclose(
        ratio,
        round(ratio),
    ):
        raise ValueError(
            "CONTROL_DT must be an integer multiple "
            "of PHYSICS_DT."
        )

    physics_steps_per_control = int(
        round(ratio)
    )

    stage_params = get_stage_parameters(
        STAGE
    )

    period = (
        2.0
        * np.pi
        / stage_params.omega
    )

    duration = (
        CYCLES
        * period
    )

    control_steps = int(
        round(
            duration
            / CONTROL_DT
        )
    )

    # ======================================================
    # Initial condition
    # ======================================================

    q0, qd0, _ = joint_reference(
        t=0.0,
        stage=STAGE,
        kinematics=kinematics,
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

    time_history = []

    ee_reference_history = []
    ee_actual_history = []

    ee_error_history = []
    joint_error_history = []

    torque_history = []

    # ======================================================
    # Closed-loop simulation
    # ======================================================

    for k in range(control_steps):

        t = (
            k
            * CONTROL_DT
        )

        q = state[:2]
        qd = state[2:]

        # --------------------------------------------------
        # Desired joint motion is obtained internally
        # from the Cartesian Stage-3 task.
        # --------------------------------------------------

        q_ref, qd_ref, qdd_ref = joint_reference(
            t=t,
            stage=STAGE,
            kinematics=kinematics,
            elbow="up",
        )

        # --------------------------------------------------
        # CTC uses the WRONG internal dynamic model.
        # --------------------------------------------------

        tau = controller.compute(
            q=q,
            qd=qd,
            q_ref=q_ref,
            qd_ref=qd_ref,
            qdd_ref=qdd_ref,
        )

        # Physical actuator limits are applied after
        # constructing the complete controller command.

        tau = np.clip(
            tau,
            -torque_limits,
            torque_limits,
        )

        # --------------------------------------------------
        # True plant integration
        #
        # Zero-order hold:
        # the same control command is held for ten
        # 500-Hz physics integration steps.
        # --------------------------------------------------

        for _ in range(
            physics_steps_per_control
        ):
            state = rk4_step(
                true_plant.state_derivative,
                state,
                PHYSICS_DT,
                tau,
            )

        # --------------------------------------------------
        # Evaluate at t + control_dt
        # --------------------------------------------------

        t_next = (
            t
            + CONTROL_DT
        )

        q_next = state[:2]

        p_actual = (
            kinematics.forward_kinematics(
                q_next
            )
        )

        p_ref, _, _ = cartesian_reference(
            t=t_next,
            stage=STAGE,
        )

        q_ref_next, _, _ = joint_reference(
            t=t_next,
            stage=STAGE,
            kinematics=kinematics,
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

        # --------------------------------------------------
        # Store
        # --------------------------------------------------

        time_history.append(
            t_next
        )

        ee_reference_history.append(
            p_ref
        )

        ee_actual_history.append(
            p_actual
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

    # ======================================================
    # Convert logs
    # ======================================================

    time_history = np.asarray(
        time_history
    )

    ee_reference_history = np.asarray(
        ee_reference_history
    )

    ee_actual_history = np.asarray(
        ee_actual_history
    )

    ee_error_history = np.asarray(
        ee_error_history
    )

    joint_error_history = np.asarray(
        joint_error_history
    )

    torque_history = np.asarray(
        torque_history
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

    ee_error_norm = np.linalg.norm(
        ee_error_history,
        axis=1,
    )

    ee_max_error = np.max(
        ee_error_norm
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
            np.abs(
                torque_history
            ),
            torque_limits,
            rtol=0.0,
            atol=1e-6,
        ),
        axis=0,
    )

    # ======================================================
    # Output directory
    # ======================================================

    output_dir = Path(
        "results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tag = "ctc_stage3_m2_70pct"

    # ======================================================
    # Text summary
    # ======================================================

    summary_path = (
        output_dir
        / f"{tag}_evaluation.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "CTC Stage 3 Model-Mismatch Evaluation\n"
        )

        f.write(
            "=====================================\n\n"
        )

        f.write(
            f"True m2 = "
            f"{true_params.m2:.6f} kg\n"
        )

        f.write(
            f"Controller m2 = "
            f"{controller_params.m2:.6f} kg\n"
        )

        f.write(
            f"Model m2 scale = "
            f"{MODEL_M2_SCALE:.6f}\n\n"
        )

        f.write(
            f"Simulation time = "
            f"{duration:.6f} s\n"
        )

        f.write(
            f"Control steps = "
            f"{control_steps}\n\n"
        )

        f.write(
            f"EE RMSE x = "
            f"{ee_rmse_xy[0]:.10f} m\n"
        )

        f.write(
            f"EE RMSE y = "
            f"{ee_rmse_xy[1]:.10f} m\n"
        )

        f.write(
            f"EE RMSE total = "
            f"{ee_rmse_total:.10f} m\n"
        )

        f.write(
            f"EE max error = "
            f"{ee_max_error:.10f} m\n\n"
        )

        f.write(
            f"Joint RMSE = "
            f"{joint_rmse.tolist()} rad\n"
        )

        f.write(
            f"Torque RMS = "
            f"{torque_rms.tolist()} Nm\n"
        )

        f.write(
            f"Torque peak = "
            f"{torque_peak.tolist()} Nm\n"
        )

        f.write(
            f"Saturation fraction = "
            f"{saturation_fraction.tolist()}\n"
        )

    # ======================================================
    # CSV
    # ======================================================

    csv_path = (
        output_dir
        / f"{tag}_trajectory.csv"
    )

    data = np.column_stack(
        [
            time_history,
            ee_reference_history[:, 0],
            ee_reference_history[:, 1],
            ee_actual_history[:, 0],
            ee_actual_history[:, 1],
            ee_error_history[:, 0],
            ee_error_history[:, 1],
            torque_history[:, 0],
            torque_history[:, 1],
        ]
    )

    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header=(
            "time,"
            "x_ref,y_ref,"
            "x_actual,y_actual,"
            "error_x,error_y,"
            "tau1,tau2"
        ),
        comments="",
    )

    # ======================================================
    # Trajectory plot
    # ======================================================

    figure_path = (
        output_dir
        / f"{tag}_trajectory.png"
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.plot(
        ee_reference_history[:, 0],
        ee_reference_history[:, 1],
        label="Reference",
        linewidth=2.0,
    )

    plt.plot(
        ee_actual_history[:, 0],
        ee_actual_history[:, 1],
        label="CTC - m2 model mismatch",
        linewidth=1.5,
    )

    plt.scatter(
        ee_reference_history[0, 0],
        ee_reference_history[0, 1],
        label="Start",
        s=50,
    )

    plt.xlabel(
        "End-effector x [m]"
    )

    plt.ylabel(
        "End-effector y [m]"
    )

    plt.title(
        "CTC Stage 3 - 30% Link-2 Mass Underestimate"
    )

    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=180,
    )

    plt.close()

    # ======================================================
    # Console output
    # ======================================================

    print()
    print("=" * 60)
    print("CTC STAGE 3 - MODEL MISMATCH")
    print("=" * 60)

    print(
        f"True m2 [kg]          : "
        f"{true_params.m2:.6f}"
    )

    print(
        f"Controller m2 [kg]    : "
        f"{controller_params.m2:.6f}"
    )

    print(
        f"Model m2 scale        : "
        f"{MODEL_M2_SCALE:.2f}"
    )

    print(
        f"Simulation time [s]   : "
        f"{duration:.3f}"
    )

    print(
        f"Control steps         : "
        f"{control_steps}"
    )

    print(
        f"EE RMSE x [m]         : "
        f"{ee_rmse_xy[0]:.10f}"
    )

    print(
        f"EE RMSE y [m]         : "
        f"{ee_rmse_xy[1]:.10f}"
    )

    print(
        f"EE RMSE total [m]     : "
        f"{ee_rmse_total:.10f}"
    )

    print(
        f"EE max error [m]      : "
        f"{ee_max_error:.10f}"
    )

    print(
        f"Joint RMSE [rad]      : "
        f"{joint_rmse}"
    )

    print(
        f"Torque RMS [Nm]       : "
        f"{torque_rms}"
    )

    print(
        f"Torque peak [Nm]      : "
        f"{torque_peak}"
    )

    print(
        f"Saturation fraction   : "
        f"{saturation_fraction}"
    )

    print()
    print(
        f"Saved: {summary_path}"
    )

    print(
        f"Saved: {csv_path}"
    )

    print(
        f"Saved: {figure_path}"
    )


if __name__ == "__main__":
    main()