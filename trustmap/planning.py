"""Shortest paths on the occupancy grid: A* and the room distance table (Point 3a).

Both simulators use these functions, so planned distances are identical in
2D and in Gazebo. The grid is a graph: each free cell is a node, and the robot
may move to a neighboring free cell.

  connectivity = 4: up, down, left, right; each move costs 1 cell.
  connectivity = 8: also the four diagonals; a diagonal move costs sqrt(2)
                    cells and is allowed only if both cells it squeezes
                    between are free (no cutting through a wall corner).

Which connectivity to use is open (OPEN_QUESTIONS Q3), so it is a required
argument. Lengths are returned in meters (cells * cell_size).

A* and Dijkstra find the same shortest length on the same grid. Dijkstra
always expands the cheapest cell not yet finished, so it grows outward from
one start cell to all cells; A* adds a guess of the remaining distance (the
heuristic, which never overestimates) so it expands fewer cells when only one
goal matters, and it returns the path itself. The precomputed table uses Dijkstra, one run per
room viewpoint; A* is used when the path geometry is needed (Gazebo goals,
plots).
"""

import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2.0)
_STRAIGHT = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAGONAL = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _check_connectivity(connectivity):
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")


def neighbors(grid, cell, connectivity):
    """Free neighbors of `cell` with their move cost in cells."""
    _check_connectivity(connectivity)
    row, col = cell
    for dr, dc in _STRAIGHT:
        nxt = (row + dr, col + dc)
        if grid.is_free(nxt):
            yield nxt, 1.0
    if connectivity == 8:
        for dr, dc in _DIAGONAL:
            nxt = (row + dr, col + dc)
            # No corner cutting: both side cells of the diagonal must be free.
            if grid.is_free(nxt) and grid.is_free((row + dr, col)) and grid.is_free((row, col + dc)):
                yield nxt, SQRT2


def heuristic(a, b, connectivity):
    """Lower bound on the remaining cells from a to b (never overestimates).

    4-connected: Manhattan distance. 8-connected: octile distance (diagonal
    moves for the shorter axis, straight moves for the rest).
    """
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    if connectivity == 4:
        return float(dr + dc)
    return (SQRT2 - 1.0) * min(dr, dc) + max(dr, dc)


def _check_cell(grid, cell, name):
    if not grid.is_free(cell):
        raise ValueError(f"{name} cell {cell} is not a free cell")


def astar(grid, start, goal, connectivity):
    """Shortest path from cell `start` to cell `goal`.

    Returns (length in meters, list of cells from start to goal), or
    (inf, []) if the goal cannot be reached.
    """
    _check_connectivity(connectivity)
    _check_cell(grid, start, "start")
    _check_cell(grid, goal, "goal")
    g = {start: 0.0}                 # best known cost from start, in cells
    parent = {start: None}
    frontier = [(heuristic(start, goal, connectivity), 0.0, start)]
    closed = set()
    while frontier:
        _, cost, cell = heapq.heappop(frontier)
        if cell in closed:
            continue  # an older, more expensive copy of this cell
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = parent[cell]
            return cost * grid.cell_size, path[::-1]
        closed.add(cell)
        for nxt, step in neighbors(grid, cell, connectivity):
            new_cost = cost + step
            if new_cost < g.get(nxt, math.inf):
                g[nxt] = new_cost
                parent[nxt] = cell
                heapq.heappush(frontier, (new_cost + heuristic(nxt, goal, connectivity), new_cost, nxt))
    return math.inf, []


def dijkstra_lengths(grid, start, connectivity, targets=None):
    """Shortest path length in meters from `start` to every cell.

    Returns a float array shaped like the grid (inf where unreachable). If
    `targets` (a list of cells) is given, the search stops as soon as all of
    them have their final length, which is much faster for the distance table.
    """
    _check_connectivity(connectivity)
    _check_cell(grid, start, "start")
    dist = np.full(grid.shape, math.inf)
    dist[start] = 0.0
    remaining = set(targets) if targets is not None else None
    frontier = [(0.0, start)]
    closed = np.zeros(grid.shape, dtype=bool)
    while frontier:
        cost, cell = heapq.heappop(frontier)
        if closed[cell]:
            continue
        closed[cell] = True
        if remaining is not None:
            remaining.discard(cell)
            if not remaining:
                break
        for nxt, step in neighbors(grid, cell, connectivity):
            new_cost = cost + step
            if new_cost < dist[nxt]:
                dist[nxt] = new_cost
                heapq.heappush(frontier, (new_cost, nxt))
    # Cells not finalized when we stopped early keep a possibly too-large
    # value; only the targets are guaranteed, so blank the rest out.
    if remaining is not None:
        dist[~closed] = math.inf
    return dist * grid.cell_size


def distance_table(grid, viewpoints, connectivity):
    """Planned distance in meters between every pair of room viewpoints (Point 3a).

    viewpoints: list of (row, col) cells, one per room, in house room order.
    Returns an n x n matrix, the `distances` of a `House`. Raises if any two
    viewpoints are not connected, since every room must be reachable.
    """
    n = len(viewpoints)
    table = np.zeros((n, n))
    for i, vp in enumerate(viewpoints):
        dist = dijkstra_lengths(grid, vp, connectivity, targets=viewpoints)
        for j, other in enumerate(viewpoints):
            table[i, j] = dist[other]
    if np.isinf(table).any():
        raise ValueError("some room viewpoints are not connected on this grid")
    # i->j and j->i are the same path backwards; averaging removes any
    # last-digit floating-point difference so the table is exactly symmetric.
    return (table + table.T) / 2.0
