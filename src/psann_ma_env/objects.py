from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .world import AgentState, GridWorld, WorldObject


@dataclass
class Food(WorldObject):
    energy: float = 0.1

    def interact(self, agent: AgentState, world: GridWorld) -> None:
        agent.energy = min(1.0, agent.energy + self.energy)


@dataclass
class Tool(WorldObject):
    name: str
    durability: int = 1

    def interact(self, agent: AgentState, world: GridWorld) -> None:
        agent.inventory[self.name] = agent.inventory.get(self.name, 0) + 1
        self.durability -= 1


@dataclass
class Deposit(WorldObject):
    target_item: Optional[str] = None
    reward: float = 0.1

    def interact(self, agent: AgentState, world: GridWorld) -> None:
        if self.target_item is None:
            return
        if agent.inventory.get(self.target_item, 0) > 0:
            agent.inventory[self.target_item] -= 1
            agent.energy = min(1.0, agent.energy + self.reward)
