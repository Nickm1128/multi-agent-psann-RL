from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np


class TerrainType(Enum):
    FLOOR = auto()
    WALL = auto()
    WATER = auto()


class Orientation(Enum):
    NORTH = auto()
    EAST = auto()
    SOUTH = auto()
    WEST = auto()

    def turn_left(self) -> "Orientation":
        turn_map = {
            Orientation.NORTH: Orientation.WEST,
            Orientation.WEST: Orientation.SOUTH,
            Orientation.SOUTH: Orientation.EAST,
            Orientation.EAST: Orientation.NORTH,
        }
        return turn_map[self]

    def turn_right(self) -> "Orientation":
        turn_map = {
            Orientation.NORTH: Orientation.EAST,
            Orientation.EAST: Orientation.SOUTH,
            Orientation.SOUTH: Orientation.WEST,
            Orientation.WEST: Orientation.NORTH,
        }
        return turn_map[self]

    @property
    def forward_delta(self) -> Tuple[int, int]:
        if self == Orientation.NORTH:
            return (-1, 0)
        if self == Orientation.SOUTH:
            return (1, 0)
        if self == Orientation.EAST:
            return (0, 1)
        return (0, -1)

    @property
    def left_delta(self) -> Tuple[int, int]:
        dr, dc = self.forward_delta
        return (-dc, dr)

    @property
    def right_delta(self) -> Tuple[int, int]:
        dr, dc = self.forward_delta
        return (dc, -dr)


@dataclass
class AgentState:
    agent_id: str
    row: int
    col: int
    orientation: Orientation
    health: float = 1.0
    energy: float = 1.0
    inventory: Dict[str, int] = field(default_factory=dict)
    role: Optional[str] = None

    def position(self) -> Tuple[int, int]:
        return (self.row, self.col)


@dataclass
class NoiseEvent:
    row: int
    col: int
    intensity: float
    symbol: Optional[int] = None


class GridWorld:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.terrain = np.full((height, width), TerrainType.FLOOR, dtype=object)
        self.objects: Dict[Tuple[int, int], List["WorldObject"]] = {}
        self.agents: Dict[str, AgentState] = {}
        self.noise_events: List[NoiseEvent] = []

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_walkable(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return False
        return self.terrain[row, col] != TerrainType.WALL

    def is_occupied(self, row: int, col: int, ignore_agent: Optional[str] = None) -> bool:
        for aid, agent in self.agents.items():
            if ignore_agent is not None and aid == ignore_agent:
                continue
            if agent.row == row and agent.col == col:
                return True
        return False

    def add_agent(self, agent: AgentState) -> None:
        self.agents[agent.agent_id] = agent

    def move_agent(self, agent_id: str, delta: Tuple[int, int]) -> None:
        agent = self.agents[agent_id]
        new_r = agent.row + delta[0]
        new_c = agent.col + delta[1]
        if self.is_walkable(new_r, new_c) and not self.is_occupied(new_r, new_c, ignore_agent=agent_id):
            agent.row, agent.col = new_r, new_c

    def add_object(self, row: int, col: int, obj: "WorldObject") -> None:
        self.objects.setdefault((row, col), []).append(obj)

    def objects_at(self, row: int, col: int) -> List["WorldObject"]:
        return self.objects.get((row, col), [])

    def record_noise(self, row: int, col: int, intensity: float, symbol: Optional[int]) -> None:
        self.noise_events.append(NoiseEvent(row=row, col=col, intensity=intensity, symbol=symbol))


class WorldObject:
    """Base class for any object placed in the grid world."""

    def interact(self, agent: AgentState, world: GridWorld) -> None:
        raise NotImplementedError

    def on_step(self, world: GridWorld) -> None:
        """Optional per-step update hook for dynamic objects."""
        return
