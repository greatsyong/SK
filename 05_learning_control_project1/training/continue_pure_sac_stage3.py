import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.two_link_cartesian_env import TwoLinkCartesianEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue Pure SAC training on Stage 3."
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing the source model and replay buffer.",
    )

    parser.add_argument(
        "--source-name",
        type=str,
        required=True,
        help="Source model basename without .zip.",
    )

    parser.add_argument(
        "--output-tag",
        type=str,
        required=True,
        help="Label used for the continuation output directory and model name.",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Additional training timesteps.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    stage = 3

    source_model = (
        args.source_dir
        / f"{args.source_name}.zip"
    )

    source_buffer = (
        args.source_dir
        / f"{args.source_name}_replay_buffer.pkl"
    )

    model_dir = Path(
        f"results/models/pure_sac_stage3_{args.output_tag}"
    )

    log_dir = Path(
        f"results/logs/pure_sac_stage3_{args.output_tag}"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Same Stage-3 task and reward semantics.
    # Replay buffer therefore remains valid.
    # ------------------------------------------------------

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

    model = SAC.load(
        str(source_model),
        env=env,
        device="auto",
        tensorboard_log=str(log_dir),
    )

    model.load_replay_buffer(
        str(source_buffer)
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(model_dir),
        name_prefix=f"sac_stage3_{args.output_tag}",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    print()
    print("=" * 60)
    print("PURE SAC CONTINUATION - STAGE 3")
    print("=" * 60)

    print("Source model        :", source_model)
    print("Source replay buffer:", source_buffer)
    print("Existing timesteps  :", model.num_timesteps)
    print("Additional steps    :", args.timesteps)
    print("Output tag          :", args.output_tag)

    print("Control rate        : 50 Hz")
    print("Physics rate        : 500 Hz")
    print("Trajectory period   : 10 s")
    print("Episode duration    : 30 s")
    print("Steps per episode   : 1500")
    print("Position scale      : 0.03 m")
    print("Replay buffer       : reused")
    print()

    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=False,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    final_name = (
        f"sac_stage3_{args.output_tag}_final"
    )

    final_path = (
        model_dir / final_name
    )

    model.save(
        str(final_path)
    )

    model.save_replay_buffer(
        str(
            model_dir
            / f"{final_name}_replay_buffer"
        )
    )

    print()
    print("=" * 60)
    print("STAGE 3 CONTINUATION COMPLETE")
    print("=" * 60)

    print("Final timesteps     :", model.num_timesteps)
    print("Final model         :", f"{final_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
