import pathlib
import sys
from copy import deepcopy

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from psann_ma_env import Action, EnvConfig, MultiAgentGridEnv  # noqa: E402
from psann_ma_env.model import PSANNWaveResNetConfig, PSANNWaveResNetQ  # noqa: E402
from psann_ma_env.rl import AgentTransition, MultiAgentDQNTrainer, ReplayBuffer  # noqa: E402


def test_psann_q_uses_conv_stem_and_fits():
    cfg = PSANNWaveResNetConfig(
        hidden_layers=1,
        hidden_units=8,
        conv_kernel_size=3,
        batch_size=4,
        epochs=1,
    )
    obs_shape = (14, 3, 3)
    q_model = PSANNWaveResNetQ(action_dim=3, obs_shape=obs_shape, psann_cfg=cfg)
    assert q_model.model.preserve_shape is True
    assert getattr(q_model.model, "_use_channel_first_train_inputs_", False)

    batch = np.random.randn(5, *obs_shape).astype(np.float32)
    targets = np.random.randn(5, 3).astype(np.float32)
    q_model.fit_q(batch, targets, verbose=0)
    preds = q_model.predict_q(batch)
    assert preds.shape == (5, 3)


def test_train_step_warm_starts_psann_models():
    env_cfg = EnvConfig(height=5, width=5, num_agents=1, vision_radius=1, audio_radius=1, max_steps=4)
    env = MultiAgentGridEnv(config=env_cfg)
    psann_cfg = PSANNWaveResNetConfig(hidden_layers=1, hidden_units=8, batch_size=2, epochs=1)
    q_model = PSANNWaveResNetQ(action_dim=len(Action), obs_shape=None, psann_cfg=psann_cfg)
    target_model = deepcopy(q_model)
    buffer = ReplayBuffer(capacity=20)
    trainer = MultiAgentDQNTrainer(
        env=env,
        q_model=q_model,
        target_model=target_model,
        buffer=buffer,
        batch_size=2,
        target_update_every=1,
    )

    obs = env.reset()
    for _ in range(2):
        actions = {aid: Action.NO_OP for aid in obs}
        next_obs, rewards, dones, _ = env.step(actions)
        for aid in obs:
            buffer.push(
                AgentTransition(
                    obs=obs[aid],
                    action=actions[aid],
                    reward=rewards[aid],
                    next_obs=next_obs[aid],
                    done=dones[aid],
                )
            )
        obs = next_obs

    # Should not raise even though neither PSANN model was pre-fitted.
    trainer.train_step()
