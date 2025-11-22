from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple
from collections import deque

import numpy as np

from .env import Action, MultiAgentGridEnv
from .model import PSANNWaveResNetQ, encode_obs_to_tensor, PSANNWaveResNetConfig


@dataclass
class AgentTransition:
    obs: Dict
    action: Action
    reward: float
    next_obs: Dict
    done: bool


class ReplayBuffer:
    """Simple replay buffer that holds per-agent transitions."""

    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer: Deque[AgentTransition] = deque(maxlen=capacity)

    def push(self, transition: AgentTransition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[AgentTransition]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class MultiAgentDQNTrainer:
    """
    Minimal multi-agent DQN trainer scaffold that feeds env observations into a PSANNRegressor-based Q network.
    This uses psann's fitting interface rather than re-implementing the backbone; suitable for small-batch updates.
    """

    def __init__(
        self,
        env: MultiAgentGridEnv,
        q_model: PSANNWaveResNetQ,
        target_model: PSANNWaveResNetQ,
        buffer: ReplayBuffer,
        gamma: float = 0.99,
        batch_size: int = 32,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 10_000,
        target_update_every: int = 500,
        log_path: Optional[str] = None,
    ) -> None:
        self.env = env
        self.q_model = q_model
        self.target_model = target_model
        self.buffer = buffer
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_every = target_update_every
        self.steps_done = 0
        self.log_path = log_path

    def select_actions(self, observations: Dict[str, Dict]) -> Dict[str, Action]:
        actions: Dict[str, Action] = {}
        epsilon = self._epsilon()
        for agent_id, obs in observations.items():
            if np.random.rand() < epsilon:
                actions[agent_id] = np.random.choice(list(Action))
            else:
                obs_tensor = encode_obs_to_tensor(obs)
                try:
                    q_values = self.q_model.predict_q(obs_tensor[None, ...])[0]
                    action_idx = int(np.argmax(q_values))
                    actions[agent_id] = list(Action)[action_idx]
                except Exception:
                    # Fallback to random if model not yet fitted.
                    actions[agent_id] = np.random.choice(list(Action))
        self.steps_done += 1
        return actions

    def _ensure_models_initialized(self, obs_batch: np.ndarray) -> None:
        """
        psann 0.12 requires calling fit before inference; warm-start the online/target
        models the first time we have a batch by fitting zeros so predict() works.
        """
        if not hasattr(self.q_model, "is_fitted"):
            return
        if not self.q_model.is_fitted():
            zeros = np.zeros((obs_batch.shape[0], self.q_model.action_dim), dtype=np.float32)
            self.q_model.fit_q(obs_batch, zeros, verbose=0)
        if hasattr(self.target_model, "is_fitted"):
            if self.target_model.is_fitted():  # type: ignore[call-arg]
                return
            self._sync_target()
        elif hasattr(self.target_model, "model"):
            self._sync_target()

    def train_step(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)
        obs_batch = np.stack([encode_obs_to_tensor(t.obs) for t in batch], axis=0)
        next_obs_batch = np.stack([encode_obs_to_tensor(t.next_obs) for t in batch], axis=0)
        actions = np.array([list(Action).index(t.action) for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        self._ensure_models_initialized(obs_batch)

        # Compute TD targets with target network.
        next_q = self.target_model.predict_q(next_obs_batch)
        next_max = np.max(next_q, axis=1)
        targets = rewards + self.gamma * (1.0 - dones) * next_max

        # Current Q predictions
        q_pred = self.q_model.predict_q(obs_batch)
        q_target_full = q_pred.copy()
        q_target_full[np.arange(self.batch_size), actions] = targets

        # Supervised fit toward TD targets using psann estimator surface.
        self.q_model.fit_q(obs_batch, q_target_full, verbose=0)

        # Periodically sync target network
        if self.steps_done % self.target_update_every == 0:
            self._sync_target()

    def _sync_target(self) -> None:
        # Deep copy psann model to target to avoid shared references.
        if hasattr(self.q_model, "is_fitted") and not self.q_model.is_fitted():
            return
        if hasattr(self.target_model, "model") and hasattr(self.q_model, "model"):
            self.target_model.model = copy.deepcopy(self.q_model.model)
        else:
            try:
                self.target_model = copy.deepcopy(self.q_model)
            except Exception:
                pass

    def _epsilon(self) -> float:
        fraction = min(1.0, self.steps_done / float(self.epsilon_decay))
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def run_episode(
        self,
        max_steps: Optional[int] = None,
        max_wall_seconds: Optional[float] = 600.0,
        *,
        render_frames: bool = False,
        renderer: Optional[Callable[[object], str]] = None,
    ) -> Dict[str, float]:
        """Run a single episode, collect transitions, and train online."""
        max_steps = max_steps or self.env.config.max_steps
        start_ts = time.time()
        obs = self.env.reset()
        total_rewards: Dict[str, float] = {aid: 0.0 for aid in obs}
        noise_counts: Dict[str, int] = {aid: 0 for aid in obs}
        interact_counts: Dict[str, int] = {aid: 0 for aid in obs}
        steps = 0
        frames: Optional[List[str]] = [] if render_frames else None
        if frames is not None:
            try:
                render_fn = renderer
                if render_fn is None:
                    from psann_ma_env.renderer import render_world as _render_world

                    render_fn = _render_world
                frames.append(render_fn(self.env.world))  # type: ignore[arg-type]
            except Exception:
                frames = None
        for _ in range(max_steps):
            if max_wall_seconds and (time.time() - start_ts) >= max_wall_seconds:
                break
            actions = self.select_actions(obs)
            next_obs, rewards, dones, _ = self.env.step(actions)
            for aid in obs:
                if actions[aid] in self.env.noise_actions:
                    noise_counts[aid] += 1
                if actions[aid] == Action.INTERACT:
                    interact_counts[aid] += 1
                self.buffer.push(
                    AgentTransition(
                        obs=obs[aid],
                        action=actions[aid],
                        reward=rewards[aid],
                        next_obs=next_obs[aid],
                        done=dones[aid],
                    )
                )
                total_rewards[aid] += rewards[aid]
            self.train_step()
            obs = next_obs
            steps += 1
            if frames is not None:
                try:
                    render_fn = renderer
                    if render_fn is None:
                        from psann_ma_env.renderer import render_world as _render_world

                        render_fn = _render_world
                    frames.append(render_fn(self.env.world))  # type: ignore[arg-type]
                except Exception:
                    frames = None
            if dones.get("__all__", False):
                break
        episode_summary = {
            "returns": total_rewards,
            "noise_counts": noise_counts,
            "interact_counts": interact_counts,
            "steps": steps,
        }
        if frames is not None:
            episode_summary["frames"] = frames
        if self.log_path:
            log_payload = {k: v for k, v in episode_summary.items() if k != "frames"}
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_payload) + "\n")
        return episode_summary

    def save_checkpoint(self, path: str) -> None:
        payload = {
            "q_model": self.q_model.model,
            "target_model": self.target_model.model,
            "buffer": list(self.buffer.buffer),
            "steps_done": self.steps_done,
        }
        with open(path, "wb") as f:
            import pickle

            pickle.dump(payload, f)

    def load_checkpoint(self, path: str) -> None:
        with open(path, "rb") as f:
            import pickle

            payload = pickle.load(f)
        self.q_model.model = payload["q_model"]
        self.target_model.model = payload["target_model"]
        self.buffer.buffer = deque(payload["buffer"], maxlen=self.buffer.capacity)
        self.steps_done = payload.get("steps_done", 0)


def build_default_trainer(env: MultiAgentGridEnv, psann_cfg: Optional[PSANNWaveResNetConfig] = None) -> MultiAgentDQNTrainer:
    from copy import deepcopy

    q_model = PSANNWaveResNetQ(action_dim=len(Action), obs_shape=None, psann_cfg=psann_cfg)  # obs_shape unused; encoded via data
    target_model = deepcopy(q_model)
    buffer = ReplayBuffer()
    return MultiAgentDQNTrainer(env=env, q_model=q_model, target_model=target_model, buffer=buffer)
