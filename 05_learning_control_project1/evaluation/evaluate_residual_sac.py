import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from stable_baselines3 import SAC

from envs.two_link_residual_cartesian_env import (
    TwoLinkResidualCartesianEnv,
)

from models.two_link_dynamics import TwoLinkDynamics

from controllers.computed_torque import (
    ComputedTorqueController,
)

from evaluation.cartesian_trajectory import (
    cartesian_reference,
    joint_reference,
)


# ==========================================================
# Fixed experiment settings
# ==========================================================

CONTROL_DT = 0.02
PHYSICS_DT = 0.002
CYCLES = 3

RESIDUAL_LIMITS = np.array(
    [5.0, 2.0],
    dtype=np.float64,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Residual SAC Cartesian tracking "
            "and compare learned residual torque against "
            "analytical model torque deficit."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Path to SAC model, with or without .zip"
        ),
    )

    parser.add_argument(
        "--tag",
        required=True,
        help=(
            "Unique experiment tag, e.g. "
            "initial, continued, continued2"
        ),
    )

    parser.add_argument(
        "--stage",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--position-scale",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--model-mass-scale",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow existing output files with the "
            "same tag to be overwritten."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    model_path = args.model
    tag = args.tag
    stage = args.stage
    position_scale = args.position_scale
    model_mass_scale = args.model_mass_scale

    # ======================================================
    # Output names
    # ======================================================

    output_dir = Path(
        "results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"residual_sac_stage{stage}_{tag}"
    )

    summary_path = (
        output_dir
        / f"{prefix}_evaluation.txt"
    )

    csv_path = (
        output_dir
        / f"{prefix}_trajectory.csv"
    )

    trajectory_path = (
        output_dir
        / f"{prefix}_trajectory.png"
    )

    deficit_path = (
        output_dir
        / (
            f"{prefix}_"
            "residual_vs_model_deficit.png"
        )
    )

    output_paths = [
        summary_path,
        csv_path,
        trajectory_path,
        deficit_path,
    ]

    # ======================================================
    # Accidental overwrite protection
    # ======================================================

    existing = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing and not args.overwrite:

        print()
        print(
            "ERROR: Output file(s) already exist:"
        )

        for path in existing:
            print(
                f"  {path}"
            )

        print()
        print(
            "Use a new --tag, or add "
            "--overwrite intentionally."
        )

        raise SystemExit(1)

    # ======================================================
    # Environment
    # ======================================================

    env = TwoLinkResidualCartesianEnv(
        stage=stage,
        control_dt=CONTROL_DT,
        physics_dt=PHYSICS_DT,
        cycles_per_episode=CYCLES,
        position_scale=position_scale,
        torque_penalty_weight=0.001,
        model_mass_scale=model_mass_scale,
        residual_limits=RESIDUAL_LIMITS,
    )

    samples_per_cycle = int(
        round(
            env.period
            / CONTROL_DT
        )
    )

    # ======================================================
    # Load trained policy
    # ======================================================

    model = SAC.load(
        model_path,
        env=env,
        device="auto",
    )

    # ======================================================
    # Exact-model CTC
    #
    # Used only for physical interpretation.
    #
    # At the SAME actual state and SAME reference:
    #
    # delta_tau_model
    # =
    # tau_true_model
    # -
    # tau_wrong_model
    #
    # ======================================================

    true_model = TwoLinkDynamics(
        env.true_params
    )

    true_ctc = ComputedTorqueController(
        model=true_model,
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
    # Reset
    # ======================================================

    obs, _ = env.reset(
        seed=42
    )

    # ======================================================
    # Logs
    # ======================================================

    time_history = []

    ee_ref_history = []
    ee_actual_history = []
    ee_error_history = []

    q_ref_history = []
    q_history = []

    action_history = []

    tau_base_history = []
    tau_residual_history = []
    tau_total_history = []

    tau_model_deficit_history = []

    reward_history = []
    saturation_history = []

    completed_full_episode = False
    terminated_final = False

    # ======================================================
    # Deterministic evaluation
    # ======================================================

    while True:

        # --------------------------------------------------
        # State before control action
        # --------------------------------------------------

        q = env.state[:2].copy()
        qd = env.state[2:].copy()

        t = env.time

        q_ref, qd_ref, qdd_ref = joint_reference(
            t=t,
            stage=stage,
            kinematics=env.kinematics,
            elbow="up",
        )

        # --------------------------------------------------
        # Exact-model CTC torque
        # --------------------------------------------------

        tau_true_model = true_ctc.compute(
            q=q,
            qd=qd,
            q_ref=q_ref,
            qd_ref=qd_ref,
            qdd_ref=qdd_ref,
        )

        # --------------------------------------------------
        # Deterministic SAC
        # --------------------------------------------------

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        # --------------------------------------------------
        # Environment step
        # --------------------------------------------------

        (
            next_obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        # --------------------------------------------------
        # Controller torque components
        # --------------------------------------------------

        tau_base = np.asarray(
            info["tau_base"],
            dtype=np.float64,
        )

        tau_residual = np.asarray(
            info["tau_residual"],
            dtype=np.float64,
        )

        tau_total = np.asarray(
            info["tau_total"],
            dtype=np.float64,
        )

        tau_total_unclipped = np.asarray(
            info["tau_total_unclipped"],
            dtype=np.float64,
        )

        # --------------------------------------------------
        # Analytical model torque deficit
        # --------------------------------------------------

        tau_model_deficit = (
            tau_true_model
            - tau_base
        )

        # --------------------------------------------------
        # State after transition
        # --------------------------------------------------

        q_next = env.state[:2].copy()

        p_actual = (
            env.kinematics.forward_kinematics(
                q_next
            )
        )

        p_ref, _, _ = cartesian_reference(
            t=env.time,
            stage=stage,
        )

        q_ref_next, _, _ = joint_reference(
            t=env.time,
            stage=stage,
            kinematics=env.kinematics,
            elbow="up",
        )

        ee_error = (
            p_ref
            - p_actual
        )

        saturated = (
            np.abs(
                tau_total_unclipped
            )
            >= env.torque_limits
        ).astype(
            np.float64
        )

        # --------------------------------------------------
        # Store
        # --------------------------------------------------

        time_history.append(
            env.time
        )

        ee_ref_history.append(
            p_ref
        )

        ee_actual_history.append(
            p_actual
        )

        ee_error_history.append(
            ee_error
        )

        q_ref_history.append(
            q_ref_next
        )

        q_history.append(
            q_next
        )

        action_history.append(
            np.asarray(
                action,
                dtype=np.float64,
            )
        )

        tau_base_history.append(
            tau_base
        )

        tau_residual_history.append(
            tau_residual
        )

        tau_total_history.append(
            tau_total
        )

        tau_model_deficit_history.append(
            tau_model_deficit
        )

        reward_history.append(
            reward
        )

        saturation_history.append(
            saturated
        )

        obs = next_obs

        if terminated or truncated:

            terminated_final = bool(
                terminated
            )

            completed_full_episode = bool(
                truncated
                and not terminated
            )

            break

    env.close()

    # ======================================================
    # Convert to arrays
    # ======================================================

    time_history = np.asarray(
        time_history
    )

    ee_ref_history = np.asarray(
        ee_ref_history
    )

    ee_actual_history = np.asarray(
        ee_actual_history
    )

    ee_error_history = np.asarray(
        ee_error_history
    )

    q_ref_history = np.asarray(
        q_ref_history
    )

    q_history = np.asarray(
        q_history
    )

    action_history = np.asarray(
        action_history
    )

    tau_base_history = np.asarray(
        tau_base_history
    )

    tau_residual_history = np.asarray(
        tau_residual_history
    )

    tau_total_history = np.asarray(
        tau_total_history
    )

    tau_model_deficit_history = np.asarray(
        tau_model_deficit_history
    )

    reward_history = np.asarray(
        reward_history
    )

    saturation_history = np.asarray(
        saturation_history
    )

    # ======================================================
    # Cartesian tracking metrics
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

    # ======================================================
    # Joint tracking metrics
    # ======================================================

    joint_error = (
        q_ref_history
        - q_history
    )

    joint_rmse = np.sqrt(
        np.mean(
            joint_error**2,
            axis=0,
        )
    )

    # ======================================================
    # Total torque metrics
    # ======================================================

    torque_rms = np.sqrt(
        np.mean(
            tau_total_history**2,
            axis=0,
        )
    )

    torque_peak = np.max(
        np.abs(
            tau_total_history
        ),
        axis=0,
    )

    saturation_fraction = np.mean(
        saturation_history,
        axis=0,
    )

    # ======================================================
    # Learned residual torque metrics
    # ======================================================

    residual_mean = np.mean(
        tau_residual_history,
        axis=0,
    )

    residual_rms = np.sqrt(
        np.mean(
            tau_residual_history**2,
            axis=0,
        )
    )

    residual_peak = np.max(
        np.abs(
            tau_residual_history
        ),
        axis=0,
    )

    # ======================================================
    # Analytical model deficit metrics
    # ======================================================

    model_deficit_mean = np.mean(
        tau_model_deficit_history,
        axis=0,
    )

    model_deficit_rms = np.sqrt(
        np.mean(
            tau_model_deficit_history**2,
            axis=0,
        )
    )

    model_deficit_peak = np.max(
        np.abs(
            tau_model_deficit_history
        ),
        axis=0,
    )

    # ======================================================
    # SAC residual vs analytical deficit
    # ======================================================

    residual_deficit_error = (
        tau_residual_history
        - tau_model_deficit_history
    )

    residual_deficit_rmse = np.sqrt(
        np.mean(
            residual_deficit_error**2,
            axis=0,
        )
    )

    residual_deficit_corr = []

    for joint in range(2):

        learned = (
            tau_residual_history[
                :,
                joint,
            ]
        )

        analytical = (
            tau_model_deficit_history[
                :,
                joint,
            ]
        )

        if (
            np.std(learned) > 1e-12
            and
            np.std(analytical) > 1e-12
        ):

            corr = np.corrcoef(
                learned,
                analytical,
            )[0, 1]

        else:
            corr = np.nan

        residual_deficit_corr.append(
            corr
        )

    residual_deficit_corr = np.asarray(
        residual_deficit_corr
    )

    # ======================================================
    # Reward
    # ======================================================

    total_reward = np.sum(
        reward_history
    )

    mean_reward = np.mean(
        reward_history
    )

    # ======================================================
    # Closed-trajectory geometric metrics
    #
    # Last complete cycle only.
    # ======================================================

    ref_last = (
        ee_ref_history[
            -samples_per_cycle:
        ]
    )

    actual_last = (
        ee_actual_history[
            -samples_per_cycle:
        ]
    )

    ref_poly = Polygon(
        ref_last
    ).buffer(0)

    actual_poly = Polygon(
        actual_last
    ).buffer(0)

    reference_area = (
        ref_poly.area
    )

    actual_area = (
        actual_poly.area
    )

    absolute_area_bias = abs(
        actual_area
        - reference_area
    )

    area_bias_percent = (
        absolute_area_bias
        / reference_area
        * 100.0
    )

    intersection_area = (
        ref_poly.intersection(
            actual_poly
        ).area
    )

    union_area = (
        ref_poly.union(
            actual_poly
        ).area
    )

    symmetric_difference_area = (
        ref_poly.symmetric_difference(
            actual_poly
        ).area
    )

    geometric_area_error_percent = (
        symmetric_difference_area
        / reference_area
        * 100.0
    )

    iou_percent = (
        intersection_area
        / union_area
        * 100.0
    )

    # ======================================================
    # Summary
    # ======================================================

    summary_lines = [

        (
            f"Residual SAC Stage {stage} "
            "Deterministic Evaluation"
        ),

        "=" * 60,
        "",

        f"Experiment tag = {tag}",
        f"Model = {model_path}",

        "",

        "Control architecture:",
        (
            "tau_total = "
            "tau_CTC_wrong + delta_tau_SAC"
        ),

        "",

        (
            f"True m2 [kg] = "
            f"{env.true_params.m2:.6f}"
        ),

        (
            f"CTC model m2 [kg] = "
            f"{env.controller_params.m2:.6f}"
        ),

        (
            "Model m2 scale = "
            f"{model_mass_scale:.6f}"
        ),

        (
            "Residual limits [Nm] = "
            f"{RESIDUAL_LIMITS.tolist()}"
        ),

        "",

        (
            "Completed full episode = "
            f"{completed_full_episode}"
        ),

        (
            f"Terminated = "
            f"{terminated_final}"
        ),

        (
            "Simulation time [s] = "
            f"{time_history[-1]:.6f}"
        ),

        (
            f"Control steps = "
            f"{len(time_history)}"
        ),

        "",

        (
            f"Total reward = "
            f"{total_reward:.10f}"
        ),

        (
            "Mean reward per step = "
            f"{mean_reward:.10f}"
        ),

        "",

        (
            "EE RMSE x [m] = "
            f"{ee_rmse_xy[0]:.10f}"
        ),

        (
            "EE RMSE y [m] = "
            f"{ee_rmse_xy[1]:.10f}"
        ),

        (
            "EE RMSE total [m] = "
            f"{ee_rmse_total:.10f}"
        ),

        (
            "EE max error [m] = "
            f"{ee_max_error:.10f}"
        ),

        "",

        (
            "Joint RMSE [rad] = "
            f"{joint_rmse.tolist()}"
        ),

        "",

        (
            "Total torque RMS [Nm] = "
            f"{torque_rms.tolist()}"
        ),

        (
            "Total torque peak [Nm] = "
            f"{torque_peak.tolist()}"
        ),

        (
            "Saturation fraction = "
            f"{saturation_fraction.tolist()}"
        ),

        "",

        (
            "Residual torque mean [Nm] = "
            f"{residual_mean.tolist()}"
        ),

        (
            "Residual torque RMS [Nm] = "
            f"{residual_rms.tolist()}"
        ),

        (
            "Residual torque peak [Nm] = "
            f"{residual_peak.tolist()}"
        ),

        "",

        (
            "Model deficit mean [Nm] = "
            f"{model_deficit_mean.tolist()}"
        ),

        (
            "Model deficit RMS [Nm] = "
            f"{model_deficit_rms.tolist()}"
        ),

        (
            "Model deficit peak [Nm] = "
            f"{model_deficit_peak.tolist()}"
        ),

        "",

        (
            "Residual-deficit RMSE [Nm] = "
            f"{residual_deficit_rmse.tolist()}"
        ),

        (
            "Residual-deficit correlation = "
            f"{residual_deficit_corr.tolist()}"
        ),

        "",

        "Geometric metrics - last complete cycle",
        "-----------------------------------------",

        (
            "Reference area [m^2] = "
            f"{reference_area:.8f}"
        ),

        (
            "Actual area [m^2] = "
            f"{actual_area:.8f}"
        ),

        (
            "Absolute area bias [m^2] = "
            f"{absolute_area_bias:.8f}"
        ),

        (
            "Area bias / reference [%] = "
            f"{area_bias_percent:.3f}"
        ),

        (
            "Symmetric-difference area [m^2] = "
            f"{symmetric_difference_area:.8f}"
        ),

        (
            "Geometric area error [%] = "
            f"{geometric_area_error_percent:.3f}"
        ),

        (
            "IoU [%] = "
            f"{iou_percent:.3f}"
        ),
    ]

    summary_text = "\n".join(
        summary_lines
    )

    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    # ======================================================
    # CSV
    # ======================================================

    csv_data = np.column_stack(
        [
            time_history,

            ee_ref_history[:, 0],
            ee_ref_history[:, 1],

            ee_actual_history[:, 0],
            ee_actual_history[:, 1],

            ee_error_history[:, 0],
            ee_error_history[:, 1],

            q_ref_history[:, 0],
            q_ref_history[:, 1],

            q_history[:, 0],
            q_history[:, 1],

            action_history[:, 0],
            action_history[:, 1],

            tau_base_history[:, 0],
            tau_base_history[:, 1],

            tau_residual_history[:, 0],
            tau_residual_history[:, 1],

            tau_total_history[:, 0],
            tau_total_history[:, 1],

            tau_model_deficit_history[:, 0],
            tau_model_deficit_history[:, 1],
        ]
    )

    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",
        comments="",
        header=(
            "time,"
            "x_ref,y_ref,"
            "x_actual,y_actual,"
            "x_error,y_error,"
            "q1_ref,q2_ref,"
            "q1,q2,"
            "action1,action2,"
            "tau_base1,tau_base2,"
            "tau_residual1,tau_residual2,"
            "tau_total1,tau_total2,"
            "tau_model_deficit1,"
            "tau_model_deficit2"
        ),
    )

    # ======================================================
    # Plot 1
    # Cartesian trajectory
    # ======================================================

    plt.figure(
        figsize=(7, 7)
    )

    plt.plot(
        ee_ref_history[:, 0],
        ee_ref_history[:, 1],
        linestyle="--",
        linewidth=2.0,
        label="Reference",
    )

    plt.plot(
        ee_actual_history[:, 0],
        ee_actual_history[:, 1],
        linewidth=1.5,
        label="Residual SAC",
    )

    plt.xlabel(
        "x [m]"
    )

    plt.ylabel(
        "y [m]"
    )

    plt.title(
        (
            f"Residual SAC Stage {stage} "
            f"- {tag}"
        )
    )

    plt.axis(
        "equal"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        trajectory_path,
        dpi=180,
    )

    plt.close()

    # ======================================================
    # Plot 2
    # SAC residual vs analytical deficit
    # ======================================================

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    axes[0].plot(
        time_history,
        tau_model_deficit_history[:, 0],
        linestyle="--",
        linewidth=2.0,
        label="Analytical model deficit",
    )

    axes[0].plot(
        time_history,
        tau_residual_history[:, 0],
        linewidth=1.3,
        label="SAC residual",
    )

    axes[0].set_ylabel(
        "Joint 1 torque [Nm]"
    )

    axes[0].grid(
        True
    )

    axes[0].legend()

    axes[1].plot(
        time_history,
        tau_model_deficit_history[:, 1],
        linestyle="--",
        linewidth=2.0,
        label="Analytical model deficit",
    )

    axes[1].plot(
        time_history,
        tau_residual_history[:, 1],
        linewidth=1.3,
        label="SAC residual",
    )

    axes[1].set_xlabel(
        "Time [s]"
    )

    axes[1].set_ylabel(
        "Joint 2 torque [Nm]"
    )

    axes[1].grid(
        True
    )

    axes[1].legend()

    fig.suptitle(
        (
            "Residual SAC vs Analytical "
            "Model Torque Deficit"
        )
    )

    fig.tight_layout()

    fig.savefig(
        deficit_path,
        dpi=180,
    )

    plt.close(
        fig
    )

    # ======================================================
    # Console
    # ======================================================

    print()
    print(
        summary_text
    )

    print()
    print(
        "Saved:"
    )

    for path in output_paths:
        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()
