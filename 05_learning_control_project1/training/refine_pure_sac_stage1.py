from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.two_link_cartesian_env import TwoLinkCartesianEnv


def main():
    # ======================================================
    # Refinement configuration
    # ======================================================

    stage = 1
    additional_timesteps = 50_000

    source_model = Path(
        "results/models/pure_sac_stage1/sac_stage1_final.zip"
    )

    model_dir = Path(
        "results/models/pure_sac_stage1_refined"
    )

    log_dir = Path(
        "results/logs/pure_sac_stage1_refined"
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
    # Environment
    # ======================================================

    env = TwoLinkCartesianEnv(
        stage=stage,
        control_dt=0.02,
        physics_dt=0.002,
        cycles_per_episode=3,

        # Stricter tracking reward than initial training.
        position_scale=0.03,

        # Keep torque penalty unchanged.
        torque_penalty_weight=0.001,
    )

    env = Monitor(
        env,
        filename=str(
            log_dir / "monitor.csv"
        ),
    )

    # ======================================================
    # Load pretrained SAC
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
    # The .zip model contains the learned network parameters,
    # but the old replay buffer is NOT loaded.
    #
    # Therefore:
    # - Actor / critic knowledge is retained.
    # - New replay experience uses the new reward definition.
    # - No old reward data are mixed into the new buffer.

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(
            model_dir
        ),
        name_prefix="sac_stage1_refined",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    # ======================================================
    # Refinement training
    # ======================================================

    print()
    print("=" * 60)
    print("PURE SAC REFINEMENT - STAGE 1")
    print("=" * 60)
    print("Source model        :", source_model)
    print("Existing timesteps  :", model.num_timesteps)
    print("Additional steps    :", additional_timesteps)
    print("Position scale      : 0.03 m")
    print("Control rate        : 50 Hz")
    print("Physics rate        : 500 Hz")
    print("Old replay buffer   : NOT loaded")
    print()

    model.learn(
        total_timesteps=additional_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,

        # Continue timestep count from pretrained model.
        # 200k -> 250k rather than resetting to zero.
        reset_num_timesteps=False,
    )

    # ======================================================
    # Save refined model
    # ======================================================

    final_path = (
        model_dir
        / "sac_stage1_refined_final"
    )

    model.save(
        str(final_path)
    )

    model.save_replay_buffer(
        str(
            model_dir
            / "sac_stage1_refined_final_replay_buffer"
        )
    )

    print()
    print("=" * 60)
    print("REFINEMENT COMPLETE")
    print("=" * 60)
    print("Final timesteps     :", model.num_timesteps)
    print(
        "Final model         :",
        f"{final_path}.zip",
    )

    env.close()


if __name__ == "__main__":
    main()
