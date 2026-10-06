from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.two_link_residual_cartesian_env import (
    TwoLinkResidualCartesianEnv,
)


def main():

    # ======================================================
    # Configuration
    # ======================================================

    additional_timesteps = 50_000

    source_dir = Path(
        "results/models/residual_sac_stage3"
    )

    source_model = (
        source_dir
        / "residual_sac_stage3_final"
    )

    source_buffer = (
        source_dir
        / "residual_sac_stage3_final_replay_buffer.pkl"
    )

    output_dir = Path(
        "results/models/residual_sac_stage3_continued"
    )

    log_dir = Path(
        "results/logs/residual_sac_stage3_continued"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ======================================================
    # Environment
    #
    # MUST remain identical to initial training.
    # ======================================================

    env = TwoLinkResidualCartesianEnv(
        stage=3,
        control_dt=0.02,
        physics_dt=0.002,
        cycles_per_episode=3,
        position_scale=0.03,
        torque_penalty_weight=0.001,
        model_mass_scale=0.70,
        residual_limits=[5.0, 2.0],
    )

    env = Monitor(
        env,
        filename=str(
            log_dir / "monitor.csv"
        ),
    )

    # ======================================================
    # Load trained model
    # ======================================================

    model = SAC.load(
        str(source_model),
        env=env,
        device="auto",
        tensorboard_log=str(
            log_dir
        ),
    )

    # ======================================================
    # Restore replay buffer
    #
    # Same task + same reward + same state/action semantics,
    # so replay reuse is valid.
    # ======================================================

    model.load_replay_buffer(
        str(source_buffer)
    )

    # ======================================================
    # Checkpoints
    # ======================================================

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(
            output_dir
        ),
        name_prefix="residual_sac_stage3_continued",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    # ======================================================
    # Summary
    # ======================================================

    print()
    print("=" * 60)
    print("RESIDUAL SAC STAGE 3 - CONTINUED TRAINING")
    print("=" * 60)

    print(
        f"Source model       : {source_model}.zip"
    )

    print(
        f"Source replay      : {source_buffer}"
    )

    print(
        "Additional steps   : "
        f"{additional_timesteps}"
    )

    print(
        "Environment        : "
        "unchanged"
    )

    print(
        "Reward semantics   : "
        "unchanged"
    )

    print(
        "Replay buffer      : "
        "reused"
    )

    print(
        "Residual limits    : "
        "[5.0, 2.0] Nm"
    )

    print()

    # ======================================================
    # Continue training
    # ======================================================

    model.learn(
        total_timesteps=additional_timesteps,
        reset_num_timesteps=False,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # ======================================================
    # Final save
    # ======================================================

    final_model = (
        output_dir
        / "residual_sac_stage3_continued_final"
    )

    final_buffer = (
        output_dir
        / "residual_sac_stage3_continued_final_replay_buffer"
    )

    model.save(
        str(final_model)
    )

    model.save_replay_buffer(
        str(final_buffer)
    )

    print()
    print("=" * 60)
    print("CONTINUED TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Cumulative timesteps :",
        model.num_timesteps,
    )

    print(
        "Final model          :",
        f"{final_model}.zip",
    )

    print(
        "Replay buffer        :",
        f"{final_buffer}.pkl",
    )

    env.close()


if __name__ == "__main__":
    main()
