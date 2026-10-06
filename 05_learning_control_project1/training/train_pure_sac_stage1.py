from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.two_link_cartesian_env import TwoLinkCartesianEnv


def main():
    # ======================================================
    # Training configuration
    # ======================================================

    stage = 1
    total_timesteps = 200_000
    seed = 42

    model_dir = Path("results/models/pure_sac_stage1")
    log_dir = Path("results/logs/pure_sac_stage1")

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ======================================================
    # Environment
    # ======================================================

    env = TwoLinkCartesianEnv(
        stage=stage,
        control_dt=0.02,
        physics_dt=0.002,
        cycles_per_episode=3,
        position_scale=0.05,
        torque_penalty_weight=0.001,
    )

    env = Monitor(
        env,
        filename=str(
            log_dir / "monitor.csv"
        ),
    )

    # ======================================================
    # SAC
    # ======================================================

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        seed=seed,
        device="auto",
        tensorboard_log=str(
            log_dir
        ),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=str(
            model_dir
        ),
        name_prefix="sac_stage1",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    # ======================================================
    # Training
    # ======================================================

    print()
    print("=" * 60)
    print("PURE SAC TRAINING - STAGE 1")
    print("=" * 60)
    print("Control rate       : 50 Hz")
    print("Physics rate       : 500 Hz")
    print("Episode duration   : 60 s")
    print("Steps per episode  : 3000")
    print("Total timesteps    :", total_timesteps)
    print("Seed               :", seed)
    print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # ======================================================
    # Save final model
    # ======================================================

    final_path = (
        model_dir
        / "sac_stage1_final"
    )

    model.save(
        str(final_path)
    )

    model.save_replay_buffer(
        str(
            model_dir
            / "sac_stage1_final_replay_buffer"
        )
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(
        "Final model:",
        f"{final_path}.zip",
    )

    env.close()


if __name__ == "__main__":
    main()
