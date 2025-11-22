Project Spec: Multi-Agent Social RL Environment with PSANN-DQN
1. High-Level Goal

Build a multi-agent reinforcement learning environment where agents, powered by a PSANN-based Deep Q-Learning architecture, can:

Perceive a local visual field (what’s in front / around them).

Perceive local audio (noises made nearby).

Take actions that include moving, interacting, and making noise.

Develop emergent social dynamics (cooperation, competition, communication) in a world that supports both simple tasks and complex, Minecraft-like behaviors over time.

The environment should be flexible enough to start with toy tasks and gradually scale up to richer, open-ended scenarios.

2. Design Principles

Multi-Agent First

From day one, the environment is built for multiple agents acting simultaneously.

Agents share the same world state but have partial observability (local vision, local audio).

Emergent Complexity, Simple Rules

The core mechanics should be small and composable (movement, resource gathering, tool use, noise/communication).

Complex behaviors (cooperation, alliances, deception, etc.) should be able to emerge from simple interactions and reward structures.

PSANN-Friendly

Q-networks should be pluggable, with a clean interface so PSANN can act as the feature extractor / backbone for the DQN policy/value network.

Reinforcement-Learning Friendly

API should be compatible with Gymnasium / PettingZoo–style environments (or a similar, clean abstraction):

reset()

step(actions)

returns: observations, rewards, done flags, info

Curriculum-Ready

It should be easy to start with:

1 agent, simple navigation/foraging

then multi-agent resource competition

then tasks that explicitly reward cooperation/communication.

3. World / Environment Overview
3.1 World Structure

Discrete 2D grid world to start (extensible later to continuous or 3D).

Each cell can contain:

Terrain type (floor, wall, water, etc.).

Resources (food, tools, objects).

Temporary markers/noise events.

Agents (one or more, depending on rules).

World size configurable (e.g., 10x10 up to 64x64+).

3.2 Objects and Resources

Basic object types:

Food / Energy items (agents must collect to survive / get reward).

Tools (can modify environment or increase efficiency of tasks).

Deposits / Goals (locations where resources must be brought).

Objects should support:

interact(agent) hooks so Codex can easily extend behaviors later.

4. Agents
4.1 Agent Properties

Each agent has:

Position (x, y).

Orientation (e.g., N/E/S/W).

Internal state:

Health / energy.

Inventory (optional: resources, tools).

Optional “role” tag (for later experiments: worker, guard, trader, etc.).

4.2 Observations

Per agent, at each step, return a structured observation:

Vision

Local egocentric view: e.g., an N x M window in front/around the agent.

Encodes:

Terrain.

Objects.

Other agents (maybe as class/type, not identity at first).

Represented as either:

Multi-channel tensor, or

Flattened feature vector.

Audio

Recent noise events in the local radius (e.g., radius R).

Could include:

Intensity.

Approximate direction.

Optional discrete “note” or “symbol” (if we want proto-language later).

Represent as a compact vector (e.g., histogram or fixed-size list of recent noises).

Self State

Scalar features:

Health/energy.

Inventory counts.

Role (one-hot).

Optional: time step / episode progress.

Combined observation should be a single feature structure easy to feed through a PSANN-based Q-network.

5. Action Space

Per agent, discrete action space including at minimum:

Movement

MOVE_FORWARD

TURN_LEFT

TURN_RIGHT

(Optionally MOVE_BACKWARD, STRAFE_LEFT, STRAFE_RIGHT in later versions)

Interaction

INTERACT (context-dependent: pick up resource, use tool, deposit item, open door, etc.)

Noise / Communication

EMIT_NOISE_k actions, where k ∈ {1..K} is a discrete symbol:

Example: EMIT_NOISE_1, EMIT_NOISE_2, ..., EMIT_NOISE_K.

These noises should:

Be logged in the environment as events.

Be perceivable by nearby agents as part of their audio observation.

Goal: allow emergent communication without imposing semantics.

No-Op

NO_OP to handle learning stability and “waiting” behavior.

6. Rewards and Task Structure

The environment should support configurable reward functions so multiple experimental setups are possible.

6.1 Base Reward Components (examples)

Survival:

Small negative reward per time step (encourages efficiency).

Penalty for running out of energy.

Resource Gathering:

+r for collecting resource.

+R for depositing resource at goal.

Cooperation / Social Dynamics (optional at first):

Shared reward for group success (e.g., everyone gets a bonus when total resources delivered exceeds a threshold).

Shaped rewards for joint actions (e.g., two agents must be present to open a gate).

Noise Use (later):

No direct reward for making noise initially, to avoid hardcoding communication.

Later, tasks can be designed where only coordinated agents (using noise) can solve.

6.2 Episodes

Finite horizon episodes:

Max steps per episode (configurable).

Episode ends if all agents die / success condition reached.

Provide hooks for:

Curriculum tasks (simple → complex).

Multi-objective rewards.

7. RL Framework & API
7.1 Environment API

Follow a multi-agent pattern similar to PettingZoo or a simple custom variant:

obs = env.reset()
for t in range(max_steps):
    actions = {agent_id: policy(obs[agent_id]) for agent_id in env.agents}
    obs, rewards, dones, infos = env.step(actions)
    if all(dones.values()):
        break


env.agents: list of active agent IDs.

Observations, rewards, and done flags are dicts keyed by agent_id.

7.2 DQN / PSANN Integration

Define a Q-network interface like:

class PSANNQNetwork(nn.Module):
    def __init__(self, obs_shape, num_actions, psann_config):
        ...

    def forward(self, obs) -> Q_values:
        ...


The deep Q-learning loop should:

Support experience replay.

Handle multi-agent transitions:

(obs[agent], action[agent], reward[agent], next_obs[agent], done[agent])

Support future extensions to centralized critics or parameter sharing among agents.

Initial version can use:

Independent Q-learning (one shared PSANN network across all agents, or per-agent networks).

Standard DQN techniques: target network, epsilon-greedy, etc.

8. Implementation Details
8.1 Language / Frameworks

Python.

Minimal dependencies for environment core (e.g., NumPy).

Optional:

Gymnasium wrappers.

PettingZoo-style wrappers.

8.2 Config and Reproducibility

Centralized config file (YAML/JSON) for:

World size, number of agents.

Observation parameters (vision radius, audio radius).

Reward coefficients.

PSANN architecture parameters.

Training hyperparameters (lr, batch size, etc.).

Logging:

Episode returns per agent.

Global stats: total resources collected, survival time, noise usage frequency, etc.

8.3 Visualization (Optional but Desired)

Simple ASCII or matplotlib-style renderer to:

Show positions of agents, resources, noise events.

Optionally record episodes for inspection.

9. Phased Roadmap (for Codex)

Phase 1: Minimal Multi-Agent Gridworld

Single environment file.

2–4 agents, local vision only (no audio yet).

Resources on the map, simple gather/deposit rewards.

Independent DQN with PSANN backbone.

Phase 2: Add Audio and Noise Actions

Implement noise events and audio observation channel.

Add noise actions (EMIT_NOISE_k).

Confirm that agents can detect nearby noises.

Phase 3: Social Tasks

Design tasks that require coordination (e.g., doors that open only when two agents are simultaneously on switches).

Use shared or partially shared rewards.

Phase 4: Emergent Communication Experiments

Analyze whether particular noise patterns correlate with behaviors (e.g., one agent “calls” others to resources).

Optionally log noise sequences for interpretability.

10. Success Criteria

Agents can learn non-trivial policies:

Efficient resource foraging and depositing.

Basic cooperation on team tasks.

The PSANN-based DQN runs stably and can be swapped in/out with minimal code changes.

The environment is modular, configurable, and can be extended with new objects, rewards, and tasks without refactoring the core.