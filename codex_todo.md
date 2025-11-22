# Codex TODO - Multi-Agent PSANN RL

## Working Rules (Must-Follow)

- Never run any command or process that is expected to take longer than 10 minutes; if something unexpectedly runs longer than 10 minutes, terminate it and record what happened in the Notes / Status section.
- Maintain and update the Notes / Status section at the bottom of this file at the end of each significant working session so that future sessions can resume without re-reading the entire codebase or spec.
- Progress through the TODO checklist from top to bottom unless there is a clear dependency or blocking issue noted; document any deviations in the Notes / Status section.
- When integrating PSANN components, align the DQN architecture with the existing `psann` docs: a PSANN `WaveResNetRegressor` combined with a per-agent liquid state machine (LSM).
- Add any new libraries to `requirements.txt` as soon as they are introduced into the codebase.

## TODO Checklist (execute in order)

### Phase 0 - Orientation & Setup

- [x] Scan the repository structure (modules, packages, scripts) and record key components and entry points in the Notes / Status section.
- [x] Identify any existing environment / RL code (Gym/PettingZoo wrappers, PSANN code, DQN training scripts) and summarize what is already implemented vs. missing.
- [x] Set up or verify a Python environment (venv/conda) and install dependencies from existing configuration files (e.g., `requirements.txt`, `pyproject.toml`) or create a minimal one if missing.
- [x] Decide on the primary environment API surface (e.g., custom multi-agent wrapper vs. PettingZoo/Gymnasium style) and write this decision in the Notes / Status section.

### Phase 1 - Environment Core

- [x] Define or validate the core grid-world state representation (terrain types, resources, agents, noise/events) in code, ensuring world size is configurable.
- [x] Implement or verify `reset()` and `step(actions)` for a multi-agent setting, including support for simultaneous actions and per-agent done flags.
- [x] Implement basic movement dynamics (forward, turning, and optional backward/strafe later) with collision rules against walls/obstacles.
- [x] Implement a generic object system with `interact(agent, world)` hooks for food, tools, deposits/goals, and any other core objects.

### Phase 2 - Observations & Audio

- [x] Implement the per-agent local egocentric visual observation (N x M window) encoding terrain, objects, and other agents into a tensor or structured feature representation.
- [x] Implement the local audio observation (recent noise events within radius R) including intensity and approximate direction, encoded as a fixed-size vector.
- [x] Implement the self-state observation vector (health/energy, inventory counts, role, optional time/episode progress) and combine all components into a single observation structure per agent.
- [x] Decide which components of agent information are part of the environment-level observation versus internal agent state (e.g., liquid state machine reservoir state, PSANN hidden state), and document this separation for the DQN architecture.
- [x] Verify that the observation structure is PSANN/PSANN-LSM-friendly (fixed shapes, consistent dtypes, and well-defined time steps) and document the exact shapes and time indexing in the Notes / Status section.

### Phase 3 - Actions, Noise, and Social Mechanics

- [x] Implement the discrete action space including movement, `INTERACT`, `EMIT_NOISE_k` symbols (for k in {1..K}), and `NO_OP`.
- [x] Ensure emitted noises are recorded as events in the world and correctly reflected in nearby agents' audio observations.
- [x] Design and implement basic rules that allow for emergent social interactions (e.g., simple resource competition, proximity effects, or interaction constraints) without hard-coding complex social behavior.

### Phase 4 - Rewards & Tasks

- [x] Implement a configurable reward function framework that can combine components such as survival cost, resource collection, delivery to goals, and simple social rewards/penalties.
- [x] Set up at least three curriculum tasks: (1) single-agent navigation/foraging, (2) multi-agent resource competition, and (3) a simple cooperative/communication-based task.
- [x] For each task, document the reward components and success criteria in the Notes / Status section.

### Phase 5 - PSANN WaveResNet-LSM DQN Integration

- [x] Define a pluggable DQN head that leverages `psann.PSANNRegressor` (WaveResNet configuration) with `lsm` preprocessor instead of reimplementing PSANN; map env observations to the expected `(batch, channels, H, W)` or `(batch, time, features)` layouts using `preserve_shape`/`per_element` and `data_format`.
- [x] Configure/use psann LSM expanders (`lsm`, `lsm_train`, `lsm_pretrain_epochs`, `lsm_lr`) to hold per-agent recurrent state; decide when to detach/reset state via psann `stateful` / `state_reset`.
- [x] Implement an explicit agent-state struct that bundles env observations + psann LSM/stateful context (kept outside env obs) as the contract to the DQN; handle init/reset on episode boundaries and detaching across steps.
- [x] Implement/adapt a multi-agent DQN loop (replay, target nets, epsilon-greedy) that feeds the psann model, manages per-agent LSM/stateful contexts, and logs metrics; reuse psann training knobs rather than rewriting optimizers/losses.
- [x] Add configuration for number of agents, observation window sizes, noise vocab size K, task selection, and psann hyperparameters (hidden layers/units, attention, WaveResNet conv params, lsm params) so experiments are reproducible and scriptable.

### Phase 6 - Evaluation, Visualization, and Documentation

- [x] Implement minimal logging and metrics (episode returns, per-agent statistics, basic social metrics like interaction counts or noise usage).
- [x] Implement a simple visualization or text-based renderer of the grid world for quick debugging (e.g., ANSI/ASCII or matplotlib-based).
- [x] Write user-facing documentation or a README section that explains how to initialize the environment, plug in a PSANN-based WaveResNet-LSM DQN, and run the curriculum tasks.
- [x] Add at least one end-to-end example script that trains agents in a simple task and logs basic results.

### Phase 7 - Final Refinements Before Running

- [x] Harden trainer: copy target network parameters (not just references), add checkpoint save/load for psann models and replay buffers.
- [x] Add richer metrics: episode length, noise usage counts, interaction counts, per-task success markers; optionally CSV/JSONL logging.
- [x] Add graceful termination rules: stop on wall-clock budget or reward plateau; respect 10-minute runtime guard.
- [x] Add CLI/config wiring: expose task preset, psann hyperparams (hidden_units/layers, attention, conv params, lsm), epsilon schedule, and seeds via argparse or config file.
- [x] Validate obs encoding: unit test `encode_obs_to_tensor` shapes/channels and reward calculations for interactions/death penalties.
- [x] Tighten example script: fix target sync to use deep copy/cloning, add simple progress bar/prints, and allow renderer snapshots for debugging.
- [ ] Optional: add plotting hook (matplotlib/text) for returns over episodes; ensure optional deps are gated.

## Notes / Status

- Date: 2025-11-16 - Phase 0-7 pass.
- Spec summary: Project is a multi-agent, partially observable 2D grid-world RL environment with local vision/audio, discrete actions (movement, interact, noise, no-op), and PSANN-based DQN agents; it should support emergent social dynamics and multiple tasks via configurable rewards. Target DQN architecture is a PSANN `WaveResNetRegressor` paired with a per-agent liquid state machine (LSM) to handle temporal structure and agent internal state.
- Repo scan: Added `src/psann_ma_env/` with `env.py`, `world.py`, `objects.py`, plus `__init__.py`; still no `psann` docs or training code.
- Existing RL/PSANN code: Newly added environment scaffolding; no training loop or PSANN/LSM integration yet.
- Environment/tooling: Created `requirements.txt` with starter deps (Gymnasium, PettingZoo, NumPy, Torch) and initialized venv at `.venv`. Add new libraries to `requirements.txt` as they are introduced.
- API decision: Target API should follow a PettingZoo/Gymnasium-style multi-agent interface (`reset`, `step(actions)`), with clear per-agent observations and action spaces, to slot into the PSANN WaveResNet-LSM DQN pipeline.
- Phase 1 outcome: Implemented grid/world state (terrain, agents, objects, noise events placeholder), movement with collision checks, multi-agent `reset`/`step` scaffold with per-agent done flags, and object interaction hooks for food/tools/deposits.
- Phase 2 outcome: Added egocentric observation builder with fixed windows: vision window size `(2*vision_radius+1)` squared with terrain (int encoded), objects (3 channels: food/tool/deposit), and agent occupancy, all rotated to align agent forward to the top. Added audio grid `(2*audio_radius+1)` squared with summed intensity of recent noise events (cleared each step) rotated egocentrically. Self-state now includes health, energy, role, orientation, time_step, inventory. Internal agent/LSM state is intentionally excluded from environment observations and will be managed within the DQN/LSM modules.
- Observation shapes/time indexing: vision terrain `int32 [H,W]`, objects `float32 [3,H,W]`, agents `float32 [H,W]`; audio `float32 [H,W]` with H=W=2*audio_radius+1; self_state scalars plus inventory dict. Step_count included for optional temporal features; noise events are cleared after each observation build.
- Phase 3 outcome: Added `EMIT_NOISE_1..3` actions (configurable up to vocab size 3) that record noise events consumed by egocentric audio. Added collision-avoidance social mechanic (agents cannot move into occupied squares) to support competition over space/resources. Maintained movement, interact, and no-op actions.
- Phase 4 outcome: Added `RewardConfig` and integrated rewards into `step` (action rewards + step penalty + death penalty). Food and deposit interactions deliver configurable rewards. Created `tasks.py` with three presets: `navigation_foraging` (single agent, focus on food), `resource_competition` (two agents, competition for food/tools/deposit), `cooperative_communication` (two agents, noise-enabled coordination for delivery). Each preset sets grid size, agent count, steps, radii, noise vocab, and reward weights.
- Task success criteria (conceptual): navigation_foraging—maximize food collection and avoid death; resource_competition—collect/deliver tools/food faster than peers under step penalties; cooperative_communication—use noise signaling to coordinate collection/delivery under step penalties with higher deposit reward.
- Phase 5 outcome: Added psann-based Q head (`PSANNWaveResNetQ`) using `psann.PSANNRegressor` with `preserve_shape` conv path, optional attention, lsm/stateful settings, and channels-first observation encoder (`encode_obs_to_tensor`). Added replay + multi-agent DQN scaffold (`rl.py`) that runs epsilon-greedy, uses psann fit/predict for TD targets, and syncs a target model. Observations are mapped to channels-first tensors with terrain/object/agent planes plus constant self-state planes; psann LSM/stateful context is kept inside the psann model.
- Phase 6 outcome: Added ASCII renderer (`renderer.py`) for quick visualization. Added minimal logging via `MultiAgentDQNTrainer.run_episode` returning per-agent returns and printing via `scripts/run_example.py`. Example script wires task presets to the trainer to demonstrate a short training loop. psann dependency recorded in `requirements.txt`. Added `ENV_USAGE.md` quickstart doc (setup, example command, tasks, renderer). Docs to consult: `Project Spec.md`, psann `README.md`, `API.md`; package exports include renderer, trainer, and model config for easier docs/usage.
- Phase 7 outcome: Trainer now deep-copies target network, supports checkpoint save/load (psann models + replay buffer), richer metrics (returns, steps, noise/interact counts) with optional JSONL logging, wall-clock guard in `run_episode`. Example script exposes CLI for epsilon schedule/logs/checkpoint/rendering. `encode_obs_to_tensor` validated for object channels. Added pytest + basic tests for env shapes, encoding, trainer run/checkpoint. Latest docs in `ENV_USAGE.md`.
- Next session starting point: optional plotting hook for returns over episodes (gated dependency), and polishing docs around checkpointing/metrics. Then dry-run longer training with 10-minute guard.

---

Shocking note: I am quietly judging every TODO you never check off.
