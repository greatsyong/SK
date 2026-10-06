import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from envs.two_link_cartesian_env import TwoLinkCartesianEnv
from evaluation.cartesian_trajectory import cartesian_reference


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--tag",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    stage = args.stage
    model_path = Path(args.model)
    tag = args.tag

    figure_path = Path(
        f"results/pure_sac_stage{stage}_{tag}_trajectory.png"
    )

    csv_path = Path(
        f"results/pure_sac_stage{stage}_{tag}_trajectory.csv"
    )

    env = TwoLinkCartesianEnv(
        stage=stage,
        control_dt=0.02,
        physics_dt=0.002,
        cycles_per_episode=3,
        position_scale=0.05,
        torque_penalty_weight=0.001,
    )

    model = SAC.load(
        str(model_path),
        env=env,
        device="auto",
    )

    obs, _ = env.reset()

    actual_history = []
    reference_history = []
    time_history = []

    terminated = False
    truncated = False

    while not (
        terminated
        or truncated
    ):

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, _, terminated, truncated, _ = env.step(
            action
        )

        p_actual = env.kin.forward_kinematics(
            env.state[:2]
        )

        p_ref, _, _ = cartesian_reference(
            t=env.time,
            stage=stage,
        )

        actual_history.append(
            p_actual.copy()
        )

        reference_history.append(
            p_ref.copy()
        )

        time_history.append(
            env.time
        )

    actual_history = np.asarray(
        actual_history
    )

    reference_history = np.asarray(
        reference_history
    )

    time_history = np.asarray(
        time_history
    )

    data = np.column_stack(
        [
            time_history,
            reference_history[:, 0],
            reference_history[:, 1],
            actual_history[:, 0],
            actual_history[:, 1],
        ]
    )

    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header="time,x_ref,y_ref,x_actual,y_actual",
        comments="",
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.plot(
        reference_history[:, 0],
        reference_history[:, 1],
        linewidth=2.0,
        label="Reference",
    )

    plt.plot(
        actual_history[:, 0],
        actual_history[:, 1],
        linewidth=1.5,
        label=f"Pure SAC - {tag}",
    )

    plt.scatter(
        reference_history[0, 0],
        reference_history[0, 1],
        s=50,
        label="Start",
    )

    plt.xlabel(
        "End-effector x [m]"
    )

    plt.ylabel(
        "End-effector y [m]"
    )

    plt.title(
        f"Pure SAC Stage {stage} - {tag}"
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
        figure_path,
        dpi=200,
    )

    print()
    print("=" * 60)
    print("PURE SAC TRAJECTORY")
    print("=" * 60)
    print("Stage   :", stage)
    print("Tag     :", tag)
    print("Model   :", model_path)
    print("Samples :", len(time_history))
    print("Figure  :", figure_path)
    print("CSV     :", csv_path)

    env.close()


if __name__ == "__main__":
    main()
