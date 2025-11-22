from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .env import EnvConfig, RewardConfig


@dataclass
class TaskSpec:
    name: str
    description: str
    env_config: EnvConfig


def task_presets() -> Dict[str, TaskSpec]:
    """Return predefined curriculum tasks with tuned configs."""
    return {
        "navigation_foraging": TaskSpec(
            name="navigation_foraging",
            description="Single-agent foraging in a small grid; focus on movement and food collection.",
            env_config=EnvConfig(
                height=8,
                width=8,
                num_agents=1,
                max_steps=200,
                vision_radius=2,
                audio_radius=2,
                noise_vocab_size=0,
                reward=RewardConfig(
                    step_penalty=-0.01,
                    food_reward=0.1,
                    deposit_reward=0.0,
                    death_penalty=-1.0,
                ),
            ),
        ),
        "resource_competition": TaskSpec(
            name="resource_competition",
            description="Two-agent competition to gather food/tools and deliver to deposit; tests spatial competition.",
            env_config=EnvConfig(
                height=10,
                width=10,
                num_agents=2,
                max_steps=300,
                vision_radius=3,
                audio_radius=3,
                noise_vocab_size=1,
                reward=RewardConfig(
                    step_penalty=-0.015,
                    food_reward=0.08,
                    deposit_reward=0.25,
                    death_penalty=-1.0,
                ),
            ),
        ),
        "cooperative_communication": TaskSpec(
            name="cooperative_communication",
            description="Two-agent cooperative task encouraging noise-based signaling and delivery coordination.",
            env_config=EnvConfig(
                height=12,
                width=12,
                num_agents=2,
                max_steps=350,
                vision_radius=3,
                audio_radius=3,
                noise_vocab_size=3,
                reward=RewardConfig(
                    step_penalty=-0.015,
                    food_reward=0.05,
                    deposit_reward=0.3,
                    death_penalty=-1.0,
                ),
            ),
        ),
    }

