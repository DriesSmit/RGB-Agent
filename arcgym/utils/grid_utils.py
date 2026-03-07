"""Grid utility functions: formatting, hashing, diffing, and component detection."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, List


# ASCII density palette: maps integer values (0-15) to characters
_ASCII_PALETTE = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1EG[]?-_+~<>i!lI;:,\"^`'. "


def format_grid_ascii(grid: List[List[int]]) -> str:
    """Format grid using ASCII density palette characters."""
    if not grid:
        return "(empty grid)"
    palette = _ASCII_PALETTE
    palette_len = len(palette)
    lines = []
    for row in grid:
        chars = []
        for v in row:
            clamped = max(0, min(15, int(v)))
            idx = min(int((clamped / 16) * (palette_len - 1)), palette_len - 1)
            chars.append(palette[idx])
        lines.append("".join(chars))
    return "\n".join(lines)


def hash_grid_state(grid: List[List[int]]) -> str:
    """Hash the full grid state."""
    return hashlib.md5(str(grid).encode()).hexdigest()[:12]


def compute_grid_diff(old_grid: list, new_grid: list) -> str:
    """Compute a compact diff between two grids, grouping by value change."""
    if not old_grid or not new_grid:
        return "(no previous state)"
    groups = defaultdict(list)
    for row_idx, (old_row, new_row) in enumerate(zip(old_grid, new_grid)):
        for col_idx, (old_val, new_val) in enumerate(zip(old_row, new_row)):
            if old_val != new_val:
                groups[(old_val, new_val)].append(f"({row_idx},{col_idx})")
    if not groups:
        return "(no change)"
    parts = []
    for (old_val, new_val), coords in sorted(groups.items()):
        parts.append(f"{old_val}->{new_val}: {', '.join(coords)}")
    return "; ".join(parts)


def find_connected_components(grid: List[List[int]]) -> Dict[tuple, int]:
    """BFS flood-fill to assign a component ID to every cell."""
    if not grid:
        return {}
    rows, cols = len(grid), len(grid[0])
    component_map: Dict[tuple, int] = {}
    component_id = 0

    def bfs(start_r: int, start_c: int, value: int) -> None:
        nonlocal component_id
        queue = [(start_r, start_c)]
        while queue:
            r, c = queue.pop(0)
            if (r, c) in component_map:
                continue
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if grid[r][c] != value:
                continue
            component_map[(r, c)] = component_id
            queue.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in component_map:
                bfs(r, c, grid[r][c])
                component_id += 1
    return component_map


def get_click_info(grid: List[List[int]], row: int, col: int) -> tuple[str, str]:
    """Return (label, component_id) for the cell at (row, col) using a single BFS.

    label: e.g. 'val=5,comp_size=12'
    component_id: e.g. 'val5_comp2'
    """
    if not grid or row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]):
        return "?", "invalid"
    value = grid[row][col]
    component_map = find_connected_components(grid)
    comp_id = component_map.get((row, col), -1)
    comp_size = sum(1 for v in component_map.values() if v == comp_id)
    return f"val={value},comp_size={comp_size}", f"val{value}_comp{comp_id}"


def get_component_label(grid: List[List[int]], row: int, col: int) -> str:
    """Return a label like 'val=5,comp_size=12' for the component at (row,col)."""
    label, _ = get_click_info(grid, row, col)
    return label


def get_component_id_for_click(grid: List[List[int]], x: int, y: int) -> str:
    """Return a string like 'val5_comp2' identifying the component at (x,y)."""
    _, comp_id = get_click_info(grid, x, y)
    return comp_id
