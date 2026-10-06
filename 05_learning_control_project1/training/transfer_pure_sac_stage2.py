from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.two_link_cartesian_env import TwoLinkCartesianEnv


def main():
    # ======================================================
    # Stage 2 configuration
    # ======================================================

    stage = 2
    additional_timesteps = 50_000

    source_model = Path(
        "results/models/pure_sac_stage1_refined/"
        "sac_stage1_refined_final.zip"
    )

    model_dir = Path(
        "results/models/pure_sac_stage2"
    )

    log_dir = Path(
        "results/logs/pure_sac_stage2"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ======================================================
    # Stage 2 environment
    # ======================================================

    env = TwoLinkCartesianEnv(
        stage=stage,
        control_dt=0.02,
        physics_dt=0.002,
        cycles_per_episode=3,
        position_scale=0.03,
        torque_penalty_weight=0.001,
    )

    env = Monitor(
        env,
        filename=str(
            log_dir / "monitor.csv"
        ),
    )

    # ======================================================
    # Transfer Stage 1 learned policy to Stage 2
    # ======================================================

    model = SAC.load(
        str(source_model),
        env=env,
        device="auto",
        tensorboard_log=str(
            log_dir
        ),
    )

    # Old replay buffer is intentionally NOT loaded.
    # Actor / critic / target networks remain pretrained.
    # New experience is collected on the Stage 2 trajectory.

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(
            model_dir
        ),
        name_prefix="sac_stage2",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    # ======================================================
    # Training
    # ======================================================

    print()
    print("=" * 60)
    print("PURE SAC TRANSFER - STAGE 1 TO STAGE 2")
    print("=" * 60)
    print("Source model        :", source_model)
    print("Existing timesteps  :", model.num_timesteps)
    print("Additional steps    :", additional_timesteps)
    print("Control rate        : 50 Hz")
    print("Physics rate        : 500 Hz")
    print("Trajectory period   : 15 s")
    print("Episode duration    : 45 s")
    print("Steps per episode   : 2250")
    print("Position scale      : 0.03 m")
    print("Old replay buffer   : NOT loaded")
    print()

    model.learn(
        total_timesteps=additional_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    # ======================================================
    # Save final Stage 2 model
    # ======================================================

    final_path = (
        model_dir
        / "sac_stage2_final"
    )

    model.save(
        str(final_path)
    )

    model.save_replay_buffer(
        str(
            model_dir
            / "sac_stage2_final_replay_buffer"
        )
    )

    print()
    print("=" * 60)
    print("STAGE 2 TRANSFER TRAINING COMPLETE")
    print("=" * 60)
    print("Final timesteps     :", model.num_timesteps)
    print(
        "Final model         :",
        f"{final_path}.zip",
    )

    env.close()


if __name__ == "__main__":
    main()
