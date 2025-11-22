import argparse
from copy import deepcopy

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


def main():
    parser = argparse.ArgumentParser(description="Run a minimal PSANN-based DQN example.")
    parser.add_argument("--task", type=str, default="navigation_foraging", choices=list(task_presets().keys()))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.1)
    parser.add_argument("--epsilon_decay", type=int, default=10000)
    parser.add_argument("--log_path", type=str, default=None, help="Optional JSONL log file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path to save at end.")
    parser.add_argument("--render_every", type=int, default=0, help="If >0, render world every N episodes.")
    args = parser.parse_args()

    preset = task_presets()[args.task]
    env = MultiAgentGridEnv(config=deepcopy(preset.env_config))

    psann_cfg = PSANNWaveResNetConfig(
        hidden_layers=2,
        hidden_units=64,
        conv_channels=64,
        conv_kernel_size=3,
        preserve_shape=True,
        per_element=False,
        data_format="channels_first",
        lsm=(
            {
                "kind": "lsmconv2dexpander",
                "out_channels": 64,
            }
            if args.task == "cooperative_communication"
            else None
        ),
        lsm_train=False,
        stateful=False,
    )

    q_model = PSANNWaveResNetQ(action_dim=len(Action), obs_shape=None, psann_cfg=psann_cfg)  # action_dim unused in init
    target_model = deepcopy(q_model)
    buffer = ReplayBuffer(capacity=5000)
    trainer = MultiAgentDQNTrainer(
        env=env,
        q_model=q_model,
        target_model=target_model,
        buffer=buffer,
        batch_size=32,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        log_path=args.log_path,
    )

    for ep in range(args.episodes):
        summary = trainer.run_episode()
        returns = summary["returns"]
        mean_return = float(np.mean(list(returns.values())))
        noise_counts = summary["noise_counts"]
        interact_counts = summary["interact_counts"]
        print(
            f"Episode {ep+1}/{args.episodes}: mean_return={mean_return:.3f}, steps={summary['steps']}, "
            f"noise={noise_counts}, interact={interact_counts}, per-agent={returns}"
        )
        if args.render_every and (ep + 1) % args.render_every == 0:
            # Render for quick debugging
            from psann_ma_env.renderer import render_world

            print(render_world(env.world))  # type: ignore

    if args.checkpoint:
        trainer.save_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
