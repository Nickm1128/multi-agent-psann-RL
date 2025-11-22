import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from psann_ma_env import Action, EnvConfig, MultiAgentGridEnv  # noqa: E402
from psann_ma_env.model import encode_obs_to_tensor  # noqa: E402


def test_env_reset_and_step_shapes():
    cfg = EnvConfig(height=6, width=6, num_agents=2, vision_radius=1, audio_radius=10, max_steps=10)
    env = MultiAgentGridEnv(config=cfg)
    obs = env.reset()
    assert len(obs) == cfg.num_agents
    for agent_obs in obs.values():
        vision = agent_obs["vision"]
        assert vision["terrain"].shape == (2 * cfg.vision_radius + 1, 2 * cfg.vision_radius + 1)
        assert vision["objects"].shape[0] == 3
        audio = agent_obs["audio"]["audio_grid"]
        assert audio.shape == (2 * cfg.audio_radius + 1, 2 * cfg.audio_radius + 1)

    actions = {aid: Action.EMIT_NOISE_1 for aid in obs}
    next_obs, rewards, dones, infos = env.step(actions)
    assert set(next_obs.keys()) == set(obs.keys())
    assert set(rewards.keys()) == set(obs.keys())
    assert "__all__" in dones


def test_encode_obs_to_tensor_channel_count():
    cfg = EnvConfig(height=6, width=6, num_agents=1, vision_radius=1, audio_radius=1, max_steps=5)
    env = MultiAgentGridEnv(config=cfg)
    obs = env.reset()
    tensor = encode_obs_to_tensor(list(obs.values())[0])
    # 3 terrain one-hot + 3 objects + 1 agents + 7 self state planes = 14
    assert tensor.shape[0] == 14
    assert tensor.dtype == np.float32
