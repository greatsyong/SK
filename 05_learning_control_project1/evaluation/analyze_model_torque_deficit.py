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
    joint_reference,
)


# ==========================================================
# Experiment configuration
# ==========================================================

STAGE = 3

CONTROL_DT = 0.02
PHYSICS_DT = 0.002
CYCLES = 3

MODEL_M2_SCALE = 0.70


def main():

    # ======================================================
    # True plant and models
    # ======================================================

    true_params = TwoLinkParams()

    wrong_params = replace(
        true_params,
        m2=true_params.m2 * MODEL_M2_SCALE,
    )

    true_plant = TwoLinkDynamics(
        true_params
    )

    true_model = TwoLinkDynamics(
        true_params
    )

    wrong_model = TwoLinkDynamics(
        wrong_params
    )

    kin = TwoLinkKinematics(
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
    # Two CTC controllers
    #
    # Same gains
    # Same measured state
    # Same reference
    #
    # Only internal dynamics model differs.
    # ======================================================

    ctc_true = ComputedTorqueController(
        model=true_model,
        kp=np.array([100.0, 100.0]),
        kd=np.array([20.0, 20.0]),
        torque_limits=None,
    )

    ctc_wrong = ComputedTorqueController(
        model=wrong_model,
        kp=np.array([100.0, 100.0]),
        kd=np.array([20.0, 20.0]),
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

    time_history = []

    tau_true_history = []
    tau_wrong_history = []
    deficit_history = []

    # ======================================================
    # Closed-loop mismatch simulation
    # ======================================================

    for k in range(control_steps):

        t = k * CONTROL_DT

        q = state[:2]
        qd = state[2:]

        q_ref, qd_ref, qdd_ref = joint_reference(
            t=t,
            stage=STAGE,
            kinematics=kin,
            elbow="up",
        )

        # --------------------------------------------------
        # Torque from exact-model CTC
        # --------------------------------------------------

        tau_true = ctc_true.compute(
            q=q,
            qd=qd,
            q_ref=q_ref,
            qd_ref=qd_ref,
            qdd_ref=qdd_ref,
        )

        # --------------------------------------------------
        # Torque from wrong-model CTC
        # --------------------------------------------------

        tau_wrong = ctc_wrong.compute(
            q=q,
            qd=qd,
            q_ref=q_ref,
            qd_ref=qd_ref,
            qdd_ref=qdd_ref,
        )

        # --------------------------------------------------
        # Analytical model torque deficit
        #
        # This is the additional torque that would convert
        # the wrong-model CTC command into the exact-model
        # CTC command at the SAME state and reference.
        # --------------------------------------------------

        delta_tau = (
            tau_true
            - tau_wrong
        )

        # --------------------------------------------------
        # Physical plant is driven by WRONG-model CTC.
        # --------------------------------------------------

        tau_applied = np.clip(
            tau_wrong,
            -torque_limits,
            torque_limits,
        )

        for _ in range(
            physics_steps_per_control
        ):
            state = rk4_step(
                true_plant.state_derivative,
                state,
                PHYSICS_DT,
                tau_applied,
            )

        # --------------------------------------------------
        # Store
        # --------------------------------------------------

        time_history.append(t)

        tau_true_history.append(
            tau_true
        )

        tau_wrong_history.append(
            tau_wrong
        )

        deficit_history.append(
            delta_tau
        )

    # ======================================================
    # Convert
    # ======================================================

    time_history = np.asarray(
        time_history
    )

    tau_true_history = np.asarray(
        tau_true_history
    )

    tau_wrong_history = np.asarray(
        tau_wrong_history
    )

    deficit_history = np.asarray(
        deficit_history
    )

    # ======================================================
    # Metrics
    # ======================================================

    deficit_rms = np.sqrt(
        np.mean(
            deficit_history**2,
            axis=0,
        )
    )

    deficit_peak = np.max(
        np.abs(
            deficit_history
        ),
        axis=0,
    )

    deficit_mean = np.mean(
        deficit_history,
        axis=0,
    )

    true_torque_rms = np.sqrt(
        np.mean(
            tau_true_history**2,
            axis=0,
        )
    )

    wrong_torque_rms = np.sqrt(
        np.mean(
            tau_wrong_history**2,
            axis=0,
        )
    )

    # ======================================================
    # Save results
    # ======================================================

    output_dir = Path("results")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tag = "stage3_m2_70pct_torque_deficit"

    # ------------------------------------------------------
    # Text
    # ------------------------------------------------------

    summary_path = (
        output_dir
        / f"{tag}.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Stage 3 Analytical Model Torque Deficit\n"
        )
        f.write(
            "=======================================\n\n"
        )

        f.write(
            f"True m2 = {true_params.m2:.6f} kg\n"
        )

        f.write(
            f"Wrong-model m2 = {wrong_params.m2:.6f} kg\n"
        )

        f.write(
            f"m2 scale = {MODEL_M2_SCALE:.6f}\n\n"
        )

        f.write(
            "Definition:\n"
        )

        f.write(
            "delta_tau_model = "
            "tau_true_model - tau_wrong_model\n\n"
        )

        f.write(
            f"Deficit mean = "
            f"{deficit_mean.tolist()} Nm\n"
        )

        f.write(
            f"Deficit RMS = "
            f"{deficit_rms.tolist()} Nm\n"
        )

        f.write(
            f"Deficit peak = "
            f"{deficit_peak.tolist()} Nm\n\n"
        )

        f.write(
            f"True-model torque RMS = "
            f"{true_torque_rms.tolist()} Nm\n"
        )

        f.write(
            f"Wrong-model torque RMS = "
            f"{wrong_torque_rms.tolist()} Nm\n"
        )

    # ------------------------------------------------------
    # CSV
    # ------------------------------------------------------

    csv_path = (
        output_dir
        / f"{tag}.csv"
    )

    data = np.column_stack(
        [
            time_history,
            tau_true_history[:, 0],
            tau_true_history[:, 1],
            tau_wrong_history[:, 0],
            tau_wrong_history[:, 1],
            deficit_history[:, 0],
            deficit_history[:, 1],
        ]
    )

    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header=(
            "time,"
            "tau_true_1,tau_true_2,"
            "tau_wrong_1,tau_wrong_2,"
            "delta_tau_1,delta_tau_2"
        ),
        comments="",
    )

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    figure_path = (
        output_dir
        / f"{tag}.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        time_history,
        deficit_history[:, 0],
        label=r"$\Delta\tau_1$",
        linewidth=1.5,
    )

    plt.plot(
        time_history,
        deficit_history[:, 1],
        label=r"$\Delta\tau_2$",
        linewidth=1.5,
    )

    plt.axhline(
        0.0,
        linewidth=1.0,
    )

    plt.xlabel(
        "Time [s]"
    )

    plt.ylabel(
        "Model torque deficit [Nm]"
    )

    plt.title(
        "Stage 3 Analytical Torque Deficit - "
        "30% Link-2 Mass Underestimate"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=180,
    )

    plt.close()

    # ======================================================
    # Console
    # ======================================================

    print()
    print("=" * 60)
    print("STAGE 3 ANALYTICAL MODEL TORQUE DEFICIT")
    print("=" * 60)

    print(
        f"True m2 [kg]          : "
        f"{true_params.m2:.6f}"
    )

    print(
        f"Wrong-model m2 [kg]   : "
        f"{wrong_params.m2:.6f}"
    )

    print(
        f"Deficit mean [Nm]     : "
        f"{deficit_mean}"
    )

    print(
        f"Deficit RMS [Nm]      : "
        f"{deficit_rms}"
    )

    print(
        f"Deficit peak [Nm]     : "
        f"{deficit_peak}"
    )

    print(
        f"True torque RMS [Nm]  : "
        f"{true_torque_rms}"
    )

    print(
        f"Wrong torque RMS [Nm] : "
        f"{wrong_torque_rms}"
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
