"""Occupancy grid shared by the 2D simulator and Gazebo (Point 3a, Agreements).

The house is stored as a 2D array of cells: True means wall or obstacle,
False means free floor. Both simulators plan and ray-cast on this same grid,
so distances and detections agree.

Layout follows ROS's nav_msgs/OccupancyGrid so the Gazebo side can convert
directly: `occupied[row, col]`, row index grows with world y, column index
grows with world x, and `origin` is the world (x, y) of the lower-left corner
of cell (0, 0).

Cell size, wall thickness and the inflation radius are still open
(docs/OPEN_QUESTIONS.md Q3), so they are arguments here, never defaults.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OccupancyGrid:
    occupied: np.ndarray       # bool array [rows, cols], True = blocked
    cell_size: float           # meters per cell side
    origin: tuple = (0.0, 0.0) # world (x, y) of the lower-left corner of cell (0, 0)

    def __post_init__(self):
        occ = np.asarray(self.occupied, dtype=bool)
        if occ.ndim != 2:
            raise ValueError("occupied must be a 2D array")
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        object.__setattr__(self, "occupied", occ)

    @property
    def shape(self):
        return self.occupied.shape

    def in_bounds(self, cell):
        row, col = cell
        rows, cols = self.shape
        return 0 <= row < rows and 0 <= col < cols

    def is_free(self, cell):
        """Free floor. Cells outside the grid count as blocked."""
        return self.in_bounds(cell) and not self.occupied[cell]

    def to_grid_coords(self, xy):
        """World (x, y) in meters -> continuous (col, row) coordinates in cell units."""
        x, y = xy
        return (x - self.origin[0]) / self.cell_size, (y - self.origin[1]) / self.cell_size

    def world_to_cell(self, xy):
        """World (x, y) -> (row, col) of the cell containing that point."""
        gx, gy = self.to_grid_coords(xy)
        return int(math.floor(gy)), int(math.floor(gx))

    def cell_to_world(self, cell):
        """(row, col) -> world (x, y) of the cell's center."""
        row, col = cell
        return (
            self.origin[0] + (col + 0.5) * self.cell_size,
            self.origin[1] + (row + 0.5) * self.cell_size,
        )


def inflate(grid, radius_m):
    """Grow every obstacle by `radius_m` so a robot of that radius fits on free cells.

    A cell becomes blocked when its center is within radius_m of the center of
    any blocked cell (a disk around each obstacle cell). Planning on the
    inflated grid lets us treat the robot as a point. radius_m = 0 returns an
    unchanged copy. The radius is open (OPEN_QUESTIONS Q3).
    """
    if radius_m < 0:
        raise ValueError("radius_m must be non-negative")
    src = grid.occupied
    out = src.copy()
    reach = int(math.floor(radius_m / grid.cell_size))
    rows, cols = src.shape
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            if (dr == 0 and dc == 0) or math.hypot(dr, dc) * grid.cell_size > radius_m:
                continue
            # Shift the obstacle mask by (dr, dc) and add it to the output.
            r_dst = slice(max(dr, 0), rows + min(dr, 0))
            c_dst = slice(max(dc, 0), cols + min(dc, 0))
            r_src = slice(max(-dr, 0), rows + min(-dr, 0))
            c_src = slice(max(-dc, 0), cols + min(-dc, 0))
            out[r_dst, c_dst] |= src[r_src, c_src]
    return OccupancyGrid(out, grid.cell_size, grid.origin)
