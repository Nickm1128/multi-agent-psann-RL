from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .world import AgentState, GridWorld, TerrainType, WorldObject
from .objects import Food, Tool, Deposit


def _object_char(objs: List[WorldObject]) -> str:
    for obj in objs:
        if isinstance(obj, Food):
            return "f"
        if isinstance(obj, Tool):
            return "t"
        if isinstance(obj, Deposit):
            return "d"
    return "."


def render_world(world: GridWorld) -> str:
    """Return an ASCII rendering of the grid world."""
    display = [[" " for _ in range(world.width)] for _ in range(world.height)]
    for r in range(world.height):
        for c in range(world.width):
            terrain = world.terrain[r, c]
            if terrain == TerrainType.WALL:
                display[r][c] = "#"
            elif terrain == TerrainType.WATER:
                display[r][c] = "~"
            else:
                display[r][c] = "."

    for (r, c), objs in world.objects.items():
        display[r][c] = _object_char(objs)

    for aid, agent in world.agents.items():
        char = aid[0].upper() if aid else "A"
        display[agent.row][agent.col] = char

    return "\n".join("".join(row) for row in display)


def _hash_color(name: str) -> Tuple[int, int, int]:
    """Deterministic pseudo-random color from a string."""
    h = sum(ord(c) for c in name) % 256
    return ((h * 37) % 256, (h * 67) % 256, (h * 97) % 256)


def render_world_image(world: GridWorld, cell_size: int = 16) -> np.ndarray:
    """Render the grid world to an RGB image array."""
    h, w = world.height, world.width
    img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)

    terrain_colors = {
        TerrainType.FLOOR: (230, 230, 230),
        TerrainType.WALL: (40, 40, 40),
        TerrainType.WATER: (70, 130, 180),
    }
    object_colors = {
        Food: (80, 180, 80),
        Tool: (255, 215, 0),
        Deposit: (200, 80, 200),
    }

    def fill_cell(r: int, c: int, color: Tuple[int, int, int]):
        img[r * cell_size : (r + 1) * cell_size, c * cell_size : (c + 1) * cell_size, :] = color

    # Base terrain
    for r in range(h):
        for c in range(w):
            color = terrain_colors.get(world.terrain[r, c], (200, 200, 200))
            fill_cell(r, c, color)

    # Objects overlay
    for (r, c), objs in world.objects.items():
        for obj in objs:
            for cls, color in object_colors.items():
                if isinstance(obj, cls):
                    fill_cell(r, c, color)
                    break

    # Noise events overlay (red tint)
    for event in world.noise_events:
        fill_cell(event.row, event.col, (220, 60, 60))

    # Agents on top
    for aid, agent in world.agents.items():
        base_color = _hash_color(aid)
        fill_cell(agent.row, agent.col, base_color)
        # Direction indicator: small stripe toward facing edge.
        r0 = agent.row * cell_size
        c0 = agent.col * cell_size
        span = max(2, cell_size // 5)
        if agent.orientation.name == "NORTH":
            img[r0 : r0 + span, c0 : c0 + cell_size, :] = 255
        elif agent.orientation.name == "SOUTH":
            img[r0 + cell_size - span : r0 + cell_size, c0 : c0 + cell_size, :] = 255
        elif agent.orientation.name == "EAST":
            img[r0 : r0 + cell_size, c0 + cell_size - span : c0 + cell_size, :] = 255
        else:
            img[r0 : r0 + cell_size, c0 : c0 + span, :] = 255

    return img
