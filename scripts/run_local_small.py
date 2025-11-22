"""
Lightweight local demo with a tiny PSANN WaveResNet Q-model to keep RAM usage low (<4 GB).
Trains across increasingly complex environments (more agents/space/noise) and saves
an animated grid render to the project root.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import imageio.v3 as iio
import numpy as np

from psann_ma_env import (
    Action,
    MultiAgentGridEnv,
    PSANNWaveResNetConfig,
    PSANNWaveResNetQ,
    ReplayBuffer,
    MultiAgentDQNTrainer,
    task_presets,
)
from psann_ma_env.model import encode_obs_to_tensor
from psann_ma_env.renderer import render_world_image


def save_animation(frames, path: Path, delay: float) -> None:
    """Persist frames to disk as a GIF (or other imageio-supported format)."""
    if not frames:
        return
    durations = [delay for _ in frames]
    iio.imwrite(path, frames, duration=durations)
    print(f"Saved animation to {path}")


def train_stage(
    *,
    trainer: MultiAgentDQNTrainer,
    env: MultiAgentGridEnv,
    episodes: int,
    max_steps: int,
) -> None:
    for ep in range(episodes):
        summary = trainer.run_episode(
            max_steps=max_steps,
            max_wall_seconds=120.0,
            render_frames=False,
        )
        returns = summary["returns"]
        mean_return = float(np.mean(list(returns.values())))
        print(
            f"Stage env {env.config.height}x{env.config.width} agents={env.config.num_agents} "
            f"episode {ep+1}/{episodes}: mean_return={mean_return:.3f}, steps={summary['steps']}, per-agent={returns}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run a tiny PSANN WaveResNet demo locally.")
    parser.add_argument("--task", type=str, default="navigation_foraging", choices=list(task_presets().keys()))
    parser.add_argument("--max_steps", type=int, default=30, help="Max steps per episode per stage.")
    parser.add_argument("--episodes_per_stage", type=int, default=2, help="Number of episodes per stage.")
    parser.add_argument("--warm_episodes", type=int, default=3, help="Warm-start episodes on a simple single-agent env.")
    parser.add_argument("--warm_max_steps", type=int, default=50, help="Max steps per warm-start episode.")
    parser.add_argument("--animate_delay", type=float, default=0.2, help="Delay between animation frames (seconds).")
    parser.add_argument("--save_animation", type=str, default="local_run.gif", help="Output animation file path.")
    parser.add_argument("--cell_size", type=int, default=16, help="Pixel size per grid cell in the render.")
    args = parser.parse_args()

    preset = task_presets()[args.task]
    stages = [
        {"height": 7, "width": 7, "num_agents": 2, "noise_vocab": 2},
        {"height": 10, "width": 10, "num_agents": 3, "noise_vocab": 3},
    ]

    psann_cfg = PSANNWaveResNetConfig(
        hidden_layers=2,
        hidden_units=64,
        conv_channels=64,
        conv_kernel_size=3,
        preserve_shape=True,
        per_element=False,
        data_format="channels_first",
        lsm={
            "kind": "lsmconv2dexpander",
            "out_channels": 64,
        },
        batch_size=8,
        epochs=1,
        lr=5e-4,
    )

    q_model = PSANNWaveResNetQ(action_dim=len(Action), obs_shape=None, psann_cfg=psann_cfg)  # action_dim unused in init

    save_path = Path(args.save_animation).resolve()

    # Warm-start on progressively more complex single-agent environments.
    warm_base_cfg = deepcopy(preset.env_config)
    warm_base_cfg.num_agents = 1
    warm_base_cfg.noise_vocab_size = 0

    # Initialise model by a tiny fit to count parameters.
    warm_env = MultiAgentGridEnv(config=warm_base_cfg)
    warm_obs = warm_env.reset()
    warm_enc = encode_obs_to_tensor(next(iter(warm_obs.values())))
    dummy_batch = np.stack([warm_enc, warm_enc], axis=0)
    dummy_targets = np.zeros((2, len(Action)), dtype=np.float32)
    q_model.fit_q(dummy_batch, dummy_targets, verbose=0)
    n_params = sum(p.numel() for p in q_model.model.model_.parameters() if p.requires_grad)  # type: ignore[attr-defined]
    print(f"Trainable parameters: {n_params}")

    warm_target_model = deepcopy(q_model)

    warm_stages = [
        {
            "height": 1,
            "width": 6,
            "vision_radius": 1,
            "audio_radius": 0,
            "max_steps": max(10, args.warm_max_steps // 2),
            "description": "1D corridor",
        },
        {
            "height": 3,
            "width": 6,
            "vision_radius": 1,
            "audio_radius": 1,
            "max_steps": args.warm_max_steps,
            "description": "strip with a second dimension",
        },
        {
            "height": 6,
            "width": 6,
            "vision_radius": 2,
            "audio_radius": 1,
            "max_steps": args.warm_max_steps,
            "description": "square starter arena",
        },
    ]

    for idx, warm_stage in enumerate(warm_stages):
        warm_cfg = deepcopy(warm_base_cfg)
        warm_cfg.height = warm_stage["height"]
        warm_cfg.width = warm_stage["width"]
        warm_cfg.vision_radius = warm_stage["vision_radius"]
        warm_cfg.audio_radius = warm_stage["audio_radius"]
        warm_cfg.max_steps = warm_stage["max_steps"]

        warm_env = MultiAgentGridEnv(config=warm_cfg)
        warm_buffer = ReplayBuffer(capacity=500)
        warm_trainer = MultiAgentDQNTrainer(
            env=warm_env,
            q_model=q_model,
            target_model=warm_target_model,
            buffer=warm_buffer,
            batch_size=8,
            epsilon_start=0.8,
            epsilon_end=0.2,
            epsilon_decay=200 + idx * 50,
            target_update_every=50,
        )
        print(
            f"Warm start stage {idx+1}/{len(warm_stages)} ({warm_stage['description']}) "
            f"on {warm_cfg.height}x{warm_cfg.width} for {args.warm_episodes} episodes."
        )
        train_stage(
            trainer=warm_trainer,
            env=warm_env,
            episodes=args.warm_episodes,
            max_steps=warm_stage["max_steps"],
        )
        q_model = warm_trainer.q_model
        warm_target_model = warm_trainer.target_model
    target_model = warm_target_model

    for idx, stage in enumerate(stages):
        env_config = deepcopy(preset.env_config)
        env_config.height = stage["height"]
        env_config.width = stage["width"]
        env_config.num_agents = stage["num_agents"]
        env_config.noise_vocab_size = stage["noise_vocab"]
        env_config.vision_radius = 2  # keep observation shape consistent and slightly richer
        env_config.audio_radius = 2
        env_config.max_steps = args.max_steps

        env = MultiAgentGridEnv(config=env_config)
        buffer = ReplayBuffer(capacity=1000)
        trainer = MultiAgentDQNTrainer(
            env=env,
            q_model=q_model,
            target_model=target_model,
            buffer=buffer,
            batch_size=8,
            epsilon_start=0.8,
            epsilon_end=0.2,
            epsilon_decay=300 + idx * 100,
            target_update_every=50,
        )

        train_stage(
            trainer=trainer,
            env=env,
            episodes=args.episodes_per_stage,
            max_steps=args.max_steps,
        )

        # Keep the latest networks for the next (more complex) stage.
        q_model = trainer.q_model
        target_model = trainer.target_model

    # After all training stages, run a final evaluation episode and save its animation.
    eval_summary = trainer.run_episode(
        max_steps=args.max_steps,
        max_wall_seconds=120.0,
        render_frames=True,
        renderer=lambda world: render_world_image(world, cell_size=args.cell_size),
    )
    if "frames" in eval_summary:
        save_animation(eval_summary["frames"], save_path, delay=args.animate_delay)


if __name__ == "__main__":
    main()
