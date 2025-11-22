import pathlib
import sys
from collections import deque

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from psann_ma_env import Action, EnvConfig, MultiAgentGridEnv  # noqa: E402
from psann_ma_env.rl import AgentTransition, MultiAgentDQNTrainer, ReplayBuffer  # noqa: E402


class DummyQ:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.model = {"dummy": True}

    def predict_q(self, obs_batch: np.ndarray) -> np.ndarray:
        batch = obs_batch.shape[0]
        # simple deterministic Q values to avoid randomness in tests
        vals = np.zeros((batch, self.action_dim), dtype=np.float32)
        vals[:, 0] = 1.0
        return vals

    def fit_q(self, obs_batch: np.ndarray, target_q: np.ndarray, verbose: int = 0) -> None:
        return


def test_trainer_run_episode_and_checkpoint(tmp_path):
    cfg = EnvConfig(height=6, width=6, num_agents=2, max_steps=5, vision_radius=1, audio_radius=2)
    env = MultiAgentGridEnv(config=cfg)
    q_model = DummyQ(action_dim=len(Action))
    target_model = DummyQ(action_dim=len(Action))
    buffer = ReplayBuffer(capacity=100)
    trainer = MultiAgentDQNTrainer(
        env=env,
        q_model=q_model,
        target_model=target_model,
        buffer=buffer,
        batch_size=4,
        target_update_every=2,
    )
    summary = trainer.run_episode(max_steps=5, max_wall_seconds=5.0)
    assert "returns" in summary and "steps" in summary
    # ensure replay buffer collected transitions
    assert len(buffer) > 0
    # checkpoint save/load
    ckpt = tmp_path / "ckpt.pkl"
    trainer.save_checkpoint(str(ckpt))
    # mutate trainer state and then load
    trainer.steps_done = 0
    trainer.buffer.buffer = deque(maxlen=buffer.capacity)
    trainer.load_checkpoint(str(ckpt))
    assert trainer.steps_done > 0
    assert len(trainer.buffer) > 0
