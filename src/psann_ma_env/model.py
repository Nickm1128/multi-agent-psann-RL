from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from psann import PSANNRegressor

from .env import Action, EnvConfig


@dataclass
class PSANNWaveResNetConfig:
    """Configuration for a PSANNRegressor used as a Q-network backbone."""

    hidden_layers: int = 3
    hidden_units: int = 128
    w0: float = 30.0
    attention: Optional[dict] = None
    conv_kernel_size: int = 3
    conv_channels: Optional[int] = None  # defaults to hidden_units if None
    preserve_shape: bool = True
    per_element: bool = False
    data_format: str = "channels_first"
    stateful: bool = False
    state_reset: str = "batch"
    lsm: Optional[dict] = None
    lsm_train: bool = False
    lsm_pretrain_epochs: int = 0
    lsm_lr: Optional[float] = None
    device: str = "auto"
    optimizer: str = "adam"
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 1  # keep small for incremental DQN-style updates


def _one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    v[i] = 1.0
    return v


def encode_obs_to_tensor(obs: Dict, num_orientation: int = 4) -> np.ndarray:
    """
    Encode environment observation dict into channels-first tensor for PSANN.
    Channels:
      - terrain one-hot (3 channels)
      - objects (3 channels: food/tool/deposit)
      - agents occupancy (1 channel)
      - self_state constant planes (health, energy, time_step_norm, orientation one-hot[4])
    """
    vision = obs["vision"]
    terrain = vision["terrain"]  # int32 [H,W] values 0..2
    objects = vision["objects"]  # float32 [3,H,W]
    agents = vision["agents"]  # float32 [H,W]
    assert objects.shape[0] == 3, "objects must have 3 channels (food, tool, deposit)"
    H, W = terrain.shape

    terrain_one_hot = np.stack(
        [
            (terrain == 0).astype(np.float32),
            (terrain == 1).astype(np.float32),
            (terrain == 2).astype(np.float32),
        ],
        axis=0,
    )
    agent_plane = agents.astype(np.float32)[None, ...]

    self_state = obs["self_state"]
    # Normalize time_step roughly by max_steps if present in obs (optional)
    time_step = float(self_state.get("time_step", 0.0))
    # Prevent division by zero; default normalizer 1.0 if not provided.
    time_norm = float(self_state.get("time_norm", 1.0)) or 1.0
    time_step_norm = time_step / time_norm

    orientation_name = self_state.get("orientation", "NORTH")
    orientation_idx = {"NORTH": 0, "EAST": 1, "SOUTH": 2, "WEST": 3}.get(orientation_name, 0)
    orientation_vec = _one_hot(orientation_idx, num_orientation)

    self_planes = [
        np.full((H, W), float(self_state.get("health", 0.0)), dtype=np.float32),
        np.full((H, W), float(self_state.get("energy", 0.0)), dtype=np.float32),
        np.full((H, W), time_step_norm, dtype=np.float32),
    ]
    for val in orientation_vec:
        self_planes.append(np.full((H, W), val, dtype=np.float32))

    self_planes = np.stack(self_planes, axis=0)

    stacked = np.concatenate([terrain_one_hot, objects, agent_plane, self_planes], axis=0)
    return stacked.astype(np.float32)


class PSANNWaveResNetQ:
    """
    Wrapper for PSANNRegressor configured as a WaveResNet-like Q-network with LSM preprocessor.
    Uses psann's built-in preserve_shape + conv parameters to avoid reimplementing the backbone.
    """

    def __init__(
        self,
        action_dim: int,
        obs_shape: Tuple[int, int, int],
        psann_cfg: Optional[PSANNWaveResNetConfig] = None,
    ) -> None:
        self.action_dim = action_dim
        self.obs_shape = obs_shape
        self.cfg = psann_cfg or PSANNWaveResNetConfig()
        conv_channels = self.cfg.conv_channels or self.cfg.hidden_units
        lsm_cfg = self.cfg.lsm or None

        self.model = PSANNRegressor.with_conv_stem(
            hidden_layers=self.cfg.hidden_layers,
            hidden_units=self.cfg.hidden_units,
            w0=self.cfg.w0,
            attention=self.cfg.attention,
            conv_kernel_size=self.cfg.conv_kernel_size,
            conv_channels=conv_channels,
            per_element=self.cfg.per_element,
            output_shape=(self.action_dim,),
            lsm=lsm_cfg,
            lsm_train=self.cfg.lsm_train,
            lsm_pretrain_epochs=self.cfg.lsm_pretrain_epochs,
            lsm_lr=self.cfg.lsm_lr,
            stateful=self.cfg.stateful,
            state_reset=self.cfg.state_reset,
            device=self.cfg.device,
            optimizer=self.cfg.optimizer,
            lr=self.cfg.lr,
            batch_size=self.cfg.batch_size,
            epochs=self.cfg.epochs,
            data_format=self.cfg.data_format,
            preserve_shape=self.cfg.preserve_shape,
        )

    def predict_q(self, obs_batch: np.ndarray) -> np.ndarray:
        """
        obs_batch: numpy array [B, C, H, W]
        Returns: Q-values [B, action_dim]
        """
        return self.model.predict(obs_batch)

    def fit_q(self, obs_batch: np.ndarray, target_q: np.ndarray, verbose: int = 0) -> None:
        """
        Supervised fit on (state, target Q) batches using psann's estimator interface.
        """
        self.model.fit(obs_batch, target_q, verbose=verbose)

    def is_fitted(self) -> bool:
        """Return True when the underlying PSANN estimator has been trained at least once."""
        return hasattr(self.model, "model_")


def build_default_q_model(env_config: EnvConfig, psann_cfg: Optional[PSANNWaveResNetConfig] = None) -> PSANNWaveResNetQ:
    h = 2 * env_config.vision_radius + 1
    w = 2 * env_config.vision_radius + 1
    # Channels: terrain one-hot (3) + objects (3) + agents (1) + self_state planes (3 + 4) = 14
    channels = 14
    action_dim = len(Action)
    return PSANNWaveResNetQ(action_dim=action_dim, obs_shape=(channels, h, w), psann_cfg=psann_cfg)
