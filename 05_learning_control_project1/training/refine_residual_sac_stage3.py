from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from envs.two_link_residual_cartesian_env import (
    TwoLinkResidualCartesianEnv,
)


# ==========================================================
# Refinement configuration
# ==========================================================

ADDITIONAL_TIMESTEPS = 30_000

ORIGINAL_LR = 3e-4
REFINEMENT_LR = 1e-4


def main():

    # ======================================================
    # Source model
    #
    # Initial 50k
    # + continued 50k
    # = 100k experience before refinement
    # ======================================================

    source_dir = Path(
        "results/models/residual_sac_stage3_continued"
    )

    source_model = (
        source_dir
        / "residual_sac_stage3_continued_final"
    )

    source_buffer = (
        source_dir
        / "residual_sac_stage3_continued_final_replay_buffer.pkl"
    )

    # ======================================================
    # Output
    # ======================================================

    output_dir = Path(
        "results/models/residual_sac_stage3_refined"
    )

    log_dir = Path(
        "results/logs/residual_sac_stage3_refined"
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
    # IMPORTANT:
    # Absolutely nothing changes here.
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
    # Load trained SAC
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
    # Same:
    # - task
    # - reward
    # - observation
    # - action semantics
    # - model mismatch
    #
    # Therefore old replay remains valid.
    # ======================================================

    model.load_replay_buffer(
        str(source_buffer)
    )

    # ======================================================
    # Learning-rate refinement
    #
    # This is the ONLY algorithmic change.
    #
    # Old:
    #     3e-4
    #
    # New:
    #     1e-4
    #
    # SAC uses the learning-rate schedule when updating
    # actor / critic / entropy-coefficient optimizers.
    #
    # Since the model was loaded from disk, replace the
    # restored schedule explicitly.
    # ======================================================

    model.learning_rate = (
        REFINEMENT_LR
    )

    model.lr_schedule = get_schedule_fn(
        REFINEMENT_LR
    )

    # ======================================================
    # Checkpoints
    # ======================================================

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(
            output_dir
        ),
        name_prefix=(
            "residual_sac_stage3_refined"
        ),
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    # ======================================================
    # Experiment summary
    # ======================================================

    print()
    print("=" * 60)
    print("RESIDUAL SAC STAGE 3 - LOW-LR REFINEMENT")
    print("=" * 60)

    print(
        f"Source model       : {source_model}.zip"
    )

    print(
        f"Source replay      : {source_buffer}"
    )

    print(
        "Previous LR        : "
        f"{ORIGINAL_LR:.1e}"
    )

    print(
        "Refinement LR      : "
        f"{REFINEMENT_LR:.1e}"
    )

    print(
        "Additional steps   : "
        f"{ADDITIONAL_TIMESTEPS}"
    )

    print(
        "Environment        : unchanged"
    )

    print(
        "Reward             : unchanged"
    )

    print(
        "Entropy settings   : unchanged"
    )

    print(
        "Network            : unchanged"
    )

    print(
        "Replay buffer      : reused"
    )

    print(
        "Residual limits    : [5.0, 2.0] Nm"
    )

    print()
    print(
        "Purpose:"
    )

    print(
        "Refine an already converged residual policy "
        "using smaller gradient-update steps."
    )

    print()

    # ======================================================
    # Continue training
    # ======================================================

    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        reset_num_timesteps=False,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # ======================================================
    # Save final refined model
    # ======================================================

    final_model = (
        output_dir
        / "residual_sac_stage3_refined_final"
    )

    final_buffer = (
        output_dir
        / (
            "residual_sac_stage3_"
            "refined_final_replay_buffer"
        )
    )

    model.save(
        str(final_model)
    )

    model.save_replay_buffer(
        str(final_buffer)
    )

    # ======================================================
    # Completion
    # ======================================================

    print()
    print("=" * 60)
    print("LOW-LR REFINEMENT COMPLETE")
    print("=" * 60)

    print(
        "Cumulative timesteps :",
        model.num_timesteps,
    )

    print(
        "Refinement LR        :",
        REFINEMENT_LR,
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
