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

    total_timesteps = 50_000
    seed = 42

    model_dir = Path(
        "results/models/residual_sac_stage3"
    )

    log_dir = Path(
        "results/logs/residual_sac_stage3"
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
    # SAC
    # ======================================================
    #
    # IMPORTANT:
    #
    # This policy is trained FROM SCRATCH.
    #
    # Pure SAC weights are intentionally not transferred
    # because the action meaning is different:
    #
    # Pure SAC:
    #     action -> full actuator torque
    #
    # Residual SAC:
    #     action -> correction torque only
    #
    # ======================================================

    model = SAC(
        policy="MlpPolicy",
        env=env,

        learning_rate=3e-4,

        buffer_size=100_000,
        learning_starts=5_000,

        batch_size=256,

        tau=0.005,
        gamma=0.99,

        train_freq=1,
        gradient_steps=1,

        ent_coef="auto",

        seed=seed,

        device="auto",

        tensorboard_log=str(
            log_dir
        ),

        verbose=1,
    )

    # ======================================================
    # Checkpoints
    # ======================================================

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(
            model_dir
        ),
        name_prefix="residual_sac_stage3",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    # ======================================================
    # Summary
    # ======================================================

    print()
    print("=" * 60)
    print("RESIDUAL SAC TRAINING - STAGE 3")
    print("=" * 60)

    print(
        "Architecture      : "
        "wrong-model CTC + SAC residual"
    )

    print(
        "True m2           : "
        "1.50 kg"
    )

    print(
        "CTC model m2      : "
        "1.05 kg"
    )

    print(
        "Residual limits   : "
        "[5.0, 2.0] Nm"
    )

    print(
        "Analytical peak   : "
        "[2.782, 0.546] Nm"
    )

    print(
        "Control rate      : "
        "50 Hz"
    )

    print(
        "Physics rate      : "
        "500 Hz"
    )

    print(
        "Episode duration  : "
        "30 s"
    )

    print(
        "Training steps    : "
        f"{total_timesteps}"
    )

    print(
        "Starting policy   : "
        "fresh SAC"
    )

    print()

    # ======================================================
    # Training
    # ======================================================

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # ======================================================
    # Final save
    # ======================================================

    final_model = (
        model_dir
        / "residual_sac_stage3_final"
    )

    final_buffer = (
        model_dir
        / "residual_sac_stage3_final_replay_buffer"
    )

    model.save(
        str(final_model)
    )

    model.save_replay_buffer(
        str(final_buffer)
    )

    print()
    print("=" * 60)
    print("RESIDUAL SAC STAGE 3 TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Timesteps         :",
        model.num_timesteps,
    )

    print(
        "Final model       :",
        f"{final_model}.zip",
    )

    print(
        "Replay buffer     :",
        f"{final_buffer}.pkl",
    )

    env.close()


if __name__ == "__main__":
    main()
