# Multi-Agent PSANN RL - Quickstart

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the example

```bash
python scripts/run_example.py --task navigation_foraging --episodes 5
```

For a very small local run (tiny WaveResNet, short episodes, low memory) that also saves an animation (`local_run.gif` by default), try:

```bash
python scripts/run_local_small.py --episodes 1 --animate
```

Available tasks (from `psann_ma_env.tasks.task_presets()`):
- `navigation_foraging` (single agent, food-focused)
- `resource_competition` (two agents, competition)
- `cooperative_communication` (two agents, noise-enabled coordination)

The example uses the psann-based WaveResNet+LSM-ready Q-head (`PSANNWaveResNetQ`) and an epsilon-greedy DQN trainer. It prints per-agent returns each episode.
It expects `psann>=0.12.0`, which requires fitting once before inference; the trainer now warm-starts
the online/target PSANN models automatically on the first batch.

## Render / debug

Use `psann_ma_env.renderer.render_world(world)` to get an ASCII map of the current grid for quick debugging.
