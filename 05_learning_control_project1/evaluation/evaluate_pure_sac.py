import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from envs.two_link_cartesian_env import TwoLinkCartesianEnv
from evaluation.cartesian_trajectory import joint_reference


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

    parser.add_argument(
        "--position-scale",
        type=float,
        required=True,
    )

    args = parser.parse_args()

    model_path = Path(args.model)

    output_path = Path(
        f"results/"
        f"pure_sac_stage{args.stage}_"
        f"{args.tag}_evaluation.txt"
    )

    env = TwoLinkCartesianEnv(
        stage=args.stage,
        control_dt=0.02,
        physics_dt=0.002,
        cycles_per_episode=3,
        position_scale=args.position_scale,
        torque_penalty_weight=0.001,
    )

    model = SAC.load(
        str(model_path),
        env=env,
        device="auto",
    )

    obs, _ = env.reset()

    ee_errors = []
    joint_errors = []
    torques = []
    rewards = []

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

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        q = env.state[:2]

        q_ref, _, _ = joint_reference(
            t=env.time,
            stage=args.stage,
            kinematics=env.kin,
            elbow="up",
        )

        ee_errors.append(
            info["ee_error"]
        )

        joint_errors.append(
            q_ref - q
        )

        torques.append(
            info["tau"]
        )

        rewards.append(
            reward
        )

    ee_errors = np.asarray(
        ee_errors,
        dtype=np.float64,
    )

    joint_errors = np.asarray(
        joint_errors,
        dtype=np.float64,
    )

    torques = np.asarray(
        torques,
        dtype=np.float64,
    )

    rewards = np.asarray(
        rewards,
        dtype=np.float64,
    )

    ee_rmse_xy = np.sqrt(
        np.mean(
            ee_errors ** 2,
            axis=0,
        )
    )

    ee_rmse_total = np.sqrt(
        np.mean(
            np.sum(
                ee_errors ** 2,
                axis=1,
            )
        )
    )

    ee_max_error = np.max(
        np.linalg.norm(
            ee_errors,
            axis=1,
        )
    )

    joint_rmse = np.sqrt(
        np.mean(
            joint_errors ** 2,
            axis=0,
        )
    )

    torque_rms = np.sqrt(
        np.mean(
            torques ** 2,
            axis=0,
        )
    )

    torque_peak = np.max(
        np.abs(
            torques
        ),
        axis=0,
    )

    saturation_fraction = np.mean(
        np.isclose(
            np.abs(torques),
            env.torque_limits,
            rtol=0.0,
            atol=1e-6,
        ),
        axis=0,
    )

    total_reward = float(
        np.sum(rewards)
    )

    mean_reward = float(
        np.mean(rewards)
    )

    completed = (
        truncated
        and not terminated
    )

    print()
    print("=" * 60)
    print(
        f"PURE SAC - STAGE {args.stage} "
        f"{args.tag.upper()} EVALUATION"
    )
    print("=" * 60)

    print("Completed full episode :", completed)
    print("Termination            :", terminated)
    print("Simulation time [s]    :", f"{env.time:.3f}")
    print("Control steps          :", env.step_count)
    print("Total reward           :", f"{total_reward:.6f}")
    print("Mean reward / step     :", f"{mean_reward:.6f}")
    print("EE RMSE x [m]          :", f"{ee_rmse_xy[0]:.10f}")
    print("EE RMSE y [m]          :", f"{ee_rmse_xy[1]:.10f}")
    print("EE RMSE total [m]      :", f"{ee_rmse_total:.10f}")
    print("EE max error [m]       :", f"{ee_max_error:.10f}")
    print("Joint RMSE [rad]       :", joint_rmse)
    print("Torque RMS [Nm]        :", torque_rms)
    print("Torque peak [Nm]       :", torque_peak)
    print("Saturation fraction    :", saturation_fraction)

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"Pure SAC Stage {args.stage} "
            f"{args.tag} Deterministic Evaluation\n"
        )
        f.write(
            "=========================================\n\n"
        )
        f.write(
            f"Model = {model_path}\n"
        )
        f.write(
            f"Position scale = {args.position_scale:.6f} m\n\n"
        )
        f.write(
            f"Completed full episode = {completed}\n"
        )
        f.write(
            f"Terminated = {terminated}\n"
        )
        f.write(
            f"Simulation time = {env.time:.6f} s\n"
        )
        f.write(
            f"Control steps = {env.step_count}\n\n"
        )
        f.write(
            f"Total reward = {total_reward:.10f}\n"
        )
        f.write(
            f"Mean reward per step = {mean_reward:.10f}\n\n"
        )
        f.write(
            f"EE RMSE x = {ee_rmse_xy[0]:.10f} m\n"
        )
        f.write(
            f"EE RMSE y = {ee_rmse_xy[1]:.10f} m\n"
        )
        f.write(
            f"EE RMSE total = {ee_rmse_total:.10f} m\n"
        )
        f.write(
            f"EE max error = {ee_max_error:.10f} m\n\n"
        )
        f.write(
            f"Joint RMSE = {joint_rmse.tolist()} rad\n"
        )
        f.write(
            f"Torque RMS = {torque_rms.tolist()} Nm\n"
        )
        f.write(
            f"Torque peak = {torque_peak.tolist()} Nm\n"
        )
        f.write(
            "Saturation fraction = "
            f"{saturation_fraction.tolist()}\n"
        )

    print()
    print("Saved:", output_path)

    env.close()


if __name__ == "__main__":
    main()
