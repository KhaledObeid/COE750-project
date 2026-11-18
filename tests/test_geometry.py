"""Occupancy grid, A*, distance table and virtual camera (Point 3a, Agreements).

All grids here are tiny hand-made test grids, not generated houses; cell size
and connectivity are test values, not decisions (OPEN_QUESTIONS Q3, Q4).
"""

import math

import numpy as np
import pytest

from trustmap.camera import cells_on_segment, in_field_of_view, line_of_sight, target_visible
from trustmap.grid import OccupancyGrid, inflate
from trustmap.planning import astar, dijkstra_lengths, distance_table


def grid_from(text, cell_size=1.0):
    """'#' = blocked, '.' = free. The first text line is the top row (largest y)."""
    lines = [line.strip() for line in text.strip().splitlines()]
    occ = np.array([[ch == "#" for ch in line] for line in reversed(lines)])
    return OccupancyGrid(occ, cell_size)


def test_world_cell_round_trip():
    g = OccupancyGrid(np.zeros((4, 6), dtype=bool), cell_size=0.5, origin=(1.0, 2.0))
    assert g.world_to_cell((1.0, 2.0)) == (0, 0)
    assert g.world_to_cell((3.74, 3.9)) == (3, 5)
    assert g.cell_to_world((3, 5)) == pytest.approx((3.75, 3.75))
    assert not g.is_free((4, 0)) and not g.is_free((0, -1))  # outside counts as blocked


def test_inflate_disk():
    occ = np.zeros((7, 7), dtype=bool)
    occ[3, 3] = True
    g = inflate(OccupancyGrid(occ, cell_size=0.1), radius_m=0.15)
    # Within 0.15 m: the 4 straight neighbors (0.1 m) and the diagonals (0.141 m), not 2 cells away.
    assert g.occupied[2:5, 2:5].all()
    assert g.occupied.sum() == 9
    assert (inflate(OccupancyGrid(occ, 0.1), 0.0).occupied == occ).all()


def test_astar_straight_and_diagonal_costs():
    g = OccupancyGrid(np.zeros((5, 5), dtype=bool), cell_size=0.5)
    length4, path4 = astar(g, (0, 0), (3, 4), connectivity=4)
    assert length4 == pytest.approx(7 * 0.5)
    assert path4[0] == (0, 0) and path4[-1] == (3, 4) and len(path4) == 8
    length8, _ = astar(g, (0, 0), (3, 4), connectivity=8)
    assert length8 == pytest.approx((3 * math.sqrt(2) + 1) * 0.5)


def test_no_corner_cutting():
    g = grid_from("""
        .#
        #.
    """)
    # The two free cells touch only at a corner between two walls.
    assert astar(g, (0, 1), (1, 0), connectivity=8)[0] == math.inf


def test_path_goes_around_wall_through_door():
    g = grid_from("""
        .......
        ###.###
        .......
    """)
    length, path = astar(g, (0, 0), (2, 0), connectivity=4)
    assert length == pytest.approx(3 + 2 + 3)
    assert (1, 3) in path


def test_astar_matches_dijkstra_on_random_grids():
    rng = np.random.default_rng(0)
    for _ in range(30):
        occ = rng.random((12, 15)) < 0.25
        occ[0, 0] = occ[11, 14] = False
        g = OccupancyGrid(occ, cell_size=0.05)
        for conn in (4, 8):
            a_len, _ = astar(g, (0, 0), (11, 14), conn)
            d_len = dijkstra_lengths(g, (0, 0), conn)[11, 14]
            assert a_len == pytest.approx(d_len) or (a_len == d_len == math.inf)


def test_distance_table_is_symmetric_with_zero_diagonal():
    g = grid_from("""
        ....#....
        ....#....
        .........
        ....#....
    """, cell_size=0.1)
    vps = [(0, 0), (3, 8), (1, 4), (0, 8)]  # (1, 4) is the gap in the wall
    table = distance_table(g, vps, connectivity=8)
    assert (table == table.T).all() and (np.diag(table) == 0).all()
    for i, a in enumerate(vps):
        for j, b in enumerate(vps):
            assert table[i, j] == pytest.approx(astar(g, a, b, 8)[0])


def test_distance_table_rejects_unreachable_room():
    g = grid_from("""
        ..#..
        ..#..
    """)
    with pytest.raises(ValueError):
        distance_table(g, [(0, 0), (0, 4)], connectivity=4)


def test_ray_cells_and_corner_rule():
    g = OccupancyGrid(np.zeros((4, 4), dtype=bool), cell_size=1.0)
    assert cells_on_segment(g, (0.5, 0.5), (3.5, 0.5)) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    # Exactly through the corner at (1, 1): both side cells are included.
    cells = cells_on_segment(g, (0.5, 0.5), (1.5, 1.5))
    assert set(cells) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_line_of_sight():
    g = grid_from("""
        .....
        ..#..
        .....
    """)
    assert not line_of_sight(g, (0.5, 1.5), (4.5, 1.5))  # wall in between
    assert line_of_sight(g, (0.5, 0.5), (4.5, 0.5))
    assert line_of_sight(g, (1.5, 1.5), (2.5, 1.5))  # target standing on the blocked cell itself


def test_field_of_view_and_range():
    pose = (0.0, 0.0, 0.0)  # facing +x
    assert in_field_of_view(pose, (1.0, 0.5), fov_rad=math.radians(90))
    assert not in_field_of_view(pose, (0.5, 1.0), fov_rad=math.radians(90))
    assert in_field_of_view(pose, (-1.0, 0.0), fov_rad=2 * math.pi)
    assert not in_field_of_view(pose, (1.6, 0.0), fov_rad=2 * math.pi)  # beyond 1.5 m


def test_target_visible_needs_both():
    g = grid_from("""
        .....
        ..#..
        .....
    """)
    pose = (0.5, 1.5, 0.0)
    assert not target_visible(g, pose, (3.5, 1.5), fov_rad=2 * math.pi, max_range_m=5.0)  # blocked
    assert target_visible(g, pose, (1.5, 0.5), fov_rad=2 * math.pi, max_range_m=5.0)
    assert not target_visible(g, (0.5, 1.5, math.pi), (1.5, 0.5), fov_rad=math.radians(60), max_range_m=5.0)
