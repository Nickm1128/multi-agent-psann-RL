from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from .objects import Deposit, Food, Tool
from .world import AgentState, GridWorld, Orientation, TerrainType


class Action(Enum):
    NO_OP = auto()
    MOVE_FORWARD = auto()
    MOVE_BACKWARD = auto()
    STRAFE_LEFT = auto()
    STRAFE_RIGHT = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    INTERACT = auto()
    EMIT_NOISE_1 = auto()
    EMIT_NOISE_2 = auto()
    EMIT_NOISE_3 = auto()


@dataclass
class EnvConfig:
    height: int = 10
    width: int = 10
    num_agents: int = 2
    max_steps: int = 200
    seed: Optional[int] = None
    vision_radius: int = 2  # visual window is (2*vision_radius + 1) squared
    audio_radius: int = 3   # audio window is (2*audio_radius + 1) squared
    noise_vocab_size: int = 3  # number of discrete noise symbols available
    reward: "RewardConfig" = field(default_factory=lambda: RewardConfig())


@dataclass
class RewardConfig:
    step_penalty: float = -0.01
    food_reward: float = 0.05
    deposit_reward: float = 0.2
    death_penalty: float = -1.0


class MultiAgentGridEnv:
    """PettingZoo/Gymnasium-style multi-agent grid environment skeleton."""

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        if self.config.noise_vocab_size > 3:
            raise ValueError("noise_vocab_size currently supports up to 3 (EMIT_NOISE_1..3).")
        self.rng = np.random.default_rng(self.config.seed)
        self.world: Optional[GridWorld] = None
        self.step_count: int = 0
        self.noise_actions = [
            Action.EMIT_NOISE_1,
            Action.EMIT_NOISE_2,
            Action.EMIT_NOISE_3,
        ][: self.config.noise_vocab_size]

    def reset(self) -> Dict[str, Dict]:
        self.world = GridWorld(height=self.config.height, width=self.config.width)
        self._validate_bounds()
        self.step_count = 0
        self._populate_world()
        return self._build_observations()

    def step(self, actions: Dict[str, Action]) -> Tuple[Dict[str, Dict], Dict[str, float], Dict[str, bool], Dict[str, Dict]]:
        assert self.world is not None, "Environment not reset."
        self.step_count += 1
        action_rewards = self._apply_actions(actions)
        self._tick_objects()
        observations = self._build_observations()
        rewards = {agent_id: action_rewards.get(agent_id, 0.0) for agent_id in self.world.agents}
        # Apply per-step survival penalty.
        for agent_id in rewards:
            rewards[agent_id] += self.config.reward.step_penalty
        dones = {agent_id: self._agent_done(agent_id) for agent_id in self.world.agents}
        # Apply terminal penalty for death/exhaustion.
        for agent_id, done in dones.items():
            if done:
                rewards[agent_id] += self.config.reward.death_penalty
        infos = {agent_id: {} for agent_id in self.world.agents}
        env_done = self.step_count >= self.config.max_steps
        dones["__all__"] = env_done or all(dones.values())
        # Decay/clear transient noise events after observation is built.
        self.world.noise_events.clear()
        return observations, rewards, dones, infos

    # Internal helpers
    def _apply_actions(self, actions: Dict[str, Action]) -> Dict[str, float]:
        reward_delta: Dict[str, float] = {aid: 0.0 for aid in self.world.agents}
        for agent_id, action in actions.items():
            if action == Action.NO_OP:
                continue
            if action == Action.TURN_LEFT:
                self.world.agents[agent_id].orientation = self.world.agents[agent_id].orientation.turn_left()
            elif action == Action.TURN_RIGHT:
                self.world.agents[agent_id].orientation = self.world.agents[agent_id].orientation.turn_right()
            elif action == Action.MOVE_FORWARD:
                delta = self.world.agents[agent_id].orientation.forward_delta
                self.world.move_agent(agent_id, delta)
            elif action == Action.MOVE_BACKWARD:
                dr, dc = self.world.agents[agent_id].orientation.forward_delta
                self.world.move_agent(agent_id, (-dr, -dc))
            elif action == Action.STRAFE_LEFT:
                delta = self.world.agents[agent_id].orientation.left_delta
                self.world.move_agent(agent_id, delta)
            elif action == Action.STRAFE_RIGHT:
                delta = self.world.agents[agent_id].orientation.right_delta
                self.world.move_agent(agent_id, delta)
            elif action == Action.INTERACT:
                agent = self.world.agents[agent_id]
                for obj in self.world.objects_at(agent.row, agent.col):
                    obj.interact(agent, self.world)
                    if isinstance(obj, Food):
                        reward_delta[agent_id] += self.config.reward.food_reward
                    elif isinstance(obj, Deposit):
                        reward_delta[agent_id] += self.config.reward.deposit_reward
            elif action in self.noise_actions:
                symbol_idx = self.noise_actions.index(action)
                self.world.record_noise(
                    row=self.world.agents[agent_id].row,
                    col=self.world.agents[agent_id].col,
                    intensity=1.0,
                    symbol=symbol_idx,
                )
        return reward_delta

    def _build_observations(self) -> Dict[str, Dict]:
        obs: Dict[str, Dict] = {}
        for agent_id, agent in self.world.agents.items():
            vision = self._build_vision(agent)
            audio = self._build_audio(agent)
            self_state = {
                "health": agent.health,
                "energy": agent.energy,
                "role": agent.role,
                "orientation": agent.orientation.name,
                "time_step": self.step_count,
                "time_norm": self.config.max_steps,
                "inventory": dict(agent.inventory),
            }
            obs[agent_id] = {
                "vision": vision,
                "audio": audio,
                "self_state": self_state,
            }
        return obs

    def _tick_objects(self) -> None:
        for objs in self.world.objects.values():
            for obj in objs:
                obj.on_step(self.world)

    def _agent_done(self, agent_id: str) -> bool:
        agent = self.world.agents[agent_id]
        return agent.health <= 0 or agent.energy <= 0

    def _populate_world(self) -> None:
        assert self.world is not None
        # Simple border walls to constrain movement.
        for r in range(self.world.height):
            for c in range(self.world.width):
                if r in (0, self.world.height - 1) or c in (0, self.world.width - 1):
                    self.world.terrain[r, c] = TerrainType.WALL

        # Place agents at random floor cells.
        for idx in range(self.config.num_agents):
            row, col = self._sample_floor_cell()
            agent_id = f"agent_{idx}"
            orientation = self.rng.choice(list(Orientation))
            self.world.add_agent(AgentState(agent_id=agent_id, row=row, col=col, orientation=orientation))

        # Drop a few starter objects.
        for _ in range(3):
            r, c = self._sample_floor_cell()
            self.world.add_object(r, c, Food())
        for _ in range(2):
            r, c = self._sample_floor_cell()
            self.world.add_object(r, c, Tool(name="basic_tool"))
        r, c = self._sample_floor_cell()
        self.world.add_object(r, c, Deposit(target_item="basic_tool"))

    def _sample_floor_cell(self) -> Tuple[int, int]:
        assert self.world is not None
        while True:
            r = self.rng.integers(1, self.world.height - 1)
            c = self.rng.integers(1, self.world.width - 1)
            if self.world.terrain[r, c] == TerrainType.FLOOR and not self.world.is_occupied(int(r), int(c)):
                return int(r), int(c)

    def _validate_bounds(self) -> None:
        """Ensure the grid has room for a walkable interior cell."""
        if self.world is None:
            return
        if self.world.height < 3 or self.world.width < 3:
            raise ValueError(
                "Environment height and width must be at least 3 to leave interior floor cells"
            )

    # Observation helpers -------------------------------------------------
    def _rotation_k(self, orientation: Orientation) -> int:
        """Return k for np.rot90 to align agent forward to the top of the window."""
        if orientation == Orientation.NORTH:
            return 0
        if orientation == Orientation.EAST:
            return -1  # rotate clockwise
        if orientation == Orientation.SOUTH:
            return 2
        return 1  # WEST

    def _build_vision(self, agent: AgentState) -> Dict[str, np.ndarray]:
        assert self.world is not None
        r = self.config.vision_radius
        size = 2 * r + 1

        terrain_encoding = {
            TerrainType.FLOOR: 0,
            TerrainType.WALL: 1,
            TerrainType.WATER: 2,
        }
        terrain = np.full((size, size), terrain_encoding[TerrainType.WALL], dtype=np.int32)
        objects = np.zeros((3, size, size), dtype=np.float32)  # channels: food, tool, deposit
        agents = np.zeros((size, size), dtype=np.float32)

        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                wr = agent.row + dr
                wc = agent.col + dc
                wr_valid = self.world.in_bounds(wr, wc)
                if wr_valid:
                    terrain[r + dr, r + dc] = terrain_encoding[self.world.terrain[wr, wc]]
                    # Object channels
                    for obj in self.world.objects_at(wr, wc):
                        if isinstance(obj, Food):
                            objects[0, r + dr, r + dc] += 1.0
                        elif isinstance(obj, Tool):
                            objects[1, r + dr, r + dc] += 1.0
                        elif isinstance(obj, Deposit):
                            objects[2, r + dr, r + dc] += 1.0
                    # Agent occupancy
                    for other_id, other in self.world.agents.items():
                        if other_id == agent.agent_id:
                            continue
                        if (other.row, other.col) == (wr, wc):
                            agents[r + dr, r + dc] += 1.0

        k = self._rotation_k(agent.orientation)
        terrain = np.rot90(terrain, k)
        objects = np.rot90(objects, axes=(1, 2), k=k)
        agents = np.rot90(agents, k)

        return {
            "terrain": terrain,
            "objects": objects,
            "agents": agents,
        }

    def _build_audio(self, agent: AgentState) -> Dict[str, np.ndarray]:
        assert self.world is not None
        r = self.config.audio_radius
        size = 2 * r + 1
        audio_grid = np.zeros((size, size), dtype=np.float32)

        for event in self.world.noise_events:
            dr = event.row - agent.row
            dc = event.col - agent.col
            if abs(dr) > r or abs(dc) > r:
                continue
            audio_grid[r + dr, r + dc] += float(event.intensity)

        k = self._rotation_k(agent.orientation)
        audio_grid = np.rot90(audio_grid, k)

        return {
            "audio_grid": audio_grid,
            "radius": r,
        }
