from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.two_link_cartesian_env import TwoLinkCartesianEnv


def main():
    # ======================================================
    # Stage 3 configuration
    # ======================================================

    stage = 3
    additional_timesteps = 50_000

    source_model = Path(
        "results/models/pure_sac_stage2_continued2/"
        "sac_stage2_continued2_final.zip"
    )

    model_dir = Path(
        "results/models/pure_sac_stage3"
    )

    log_dir = Path(
        "results/logs/pure_sac_stage3"
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
    # Stage 3 environment
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
    # Transfer Stage 2 policy to Stage 3
    # ======================================================

    model = SAC.load(
        str(source_model),
        env=env,
        device="auto",
        tensorboard_log=str(
            log_dir
        ),
    )

    # IMPORTANT:
    # Stage 3 is a new task distribution.
    # The pretrained network weights are retained,
    # but the Stage 2 replay buffer is NOT loaded.

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(
            model_dir
        ),
        name_prefix="sac_stage3",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    # ======================================================
    # Training
    # ======================================================

    print()
    print("=" * 60)
    print("PURE SAC TRANSFER - STAGE 2 TO STAGE 3")
    print("=" * 60)
    print("Source model        :", source_model)
    print("Existing timesteps  :", model.num_timesteps)
    print("Additional steps    :", additional_timesteps)
    print("Control rate        : 50 Hz")
    print("Physics rate        : 500 Hz")
    print("Trajectory period   : 10 s")
    print("Episode duration    : 30 s")
    print("Steps per episode   : 1500")
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
    # Save Stage 3 model
    # ======================================================

    final_path = (
        model_dir
        / "sac_stage3_final"
    )

    model.save(
        str(final_path)
    )

    model.save_replay_buffer(
        str(
            model_dir
            / "sac_stage3_final_replay_buffer"
        )
    )

    print()
    print("=" * 60)
    print("STAGE 3 TRANSFER TRAINING COMPLETE")
    print("=" * 60)
    print("Final timesteps     :", model.num_timesteps)
    print(
        "Final model         :",
        f"{final_path}.zip",
    )

    env.close()


if __name__ == "__main__":
    main()
