"""House generator (Point 3a, engineering parameters of 2026-11-18)."""

import collections

import numpy as np
import pytest

from trustmap.camera import target_visible
from trustmap.generator import (
    CONNECTIVITY,
    DOOR_WIDTH_M,
    MIN_ROOM_WIDTH_M,
    PUBLIC_TYPES,
    ROBOT_RADIUS_M,
    generate_house,
    object_candidates,
    place_object,
)
from trustmap.planning import astar

SEEDS = range(40)


@pytest.fixture(scope="module")
def layouts():
    return [generate_house(s) for s in SEEDS]


def test_same_seed_same_house():
    a, b = generate_house(7), generate_house(7)
    assert a.rooms == b.rooms and a.doors == b.doors
    assert np.array_equal(a.house.distances, b.house.distances)


def test_footprint_and_room_sizes(layouts):
    for L in layouts:
        assert 8.0 <= L.width_m <= 12.0 and 8.0 <= L.depth_m <= 12.0
        assert len(L.rooms) == 5
        for r in L.rooms:
            x0, y0, x1, y1 = r.rect
            assert min(x1 - x0, y1 - y0) * L.grid.cell_size >= MIN_ROOM_WIDTH_M - 1e-9


def test_zones_are_on_separate_sides(layouts):
    # The first cut separates public from private: no private room lies between public rooms.
    for L in layouts:
        horizontal_cut = L.width_m >= L.depth_m
        axis = 0 if horizontal_cut else 1
        public_far = max(r.rect[axis + 2] for r in L.rooms if r.type in PUBLIC_TYPES)
        private_near = min(r.rect[axis] for r in L.rooms if r.type not in PUBLIC_TYPES)
        assert public_far == private_near


def test_doors_form_a_tree_with_one_bathroom_door(layouts):
    for L in layouts:
        assert len(L.doors) == 4  # 5 rooms, a tree has 4 links
        parent = list(range(5))

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i

        for d in L.doors:
            a, b = d.rooms
            parent[find(a)] = find(b)
        assert len({find(i) for i in range(5)}) == 1  # connected, so it is a tree
        bath = L.rooms.index(next(r for r in L.rooms if r.type == "bathroom"))
        assert sum(bath in d.rooms for d in L.doors) == 1
        for d in L.doors:  # nothing is reached through the bathroom
            if bath in d.rooms:
                other = d.rooms[0] if d.rooms[1] == bath else d.rooms[1]
                assert sum(other in e.rooms for e in L.doors) >= 1


def test_door_opening_widths(layouts):
    m = 2  # grid margin in cells
    for L in layouts:
        cs = L.grid.cell_size
        for d in L.doors:
            # Free cells along the wall line, from 6 cells before to 6 cells after the
            # door centre; this stays on the shared wall (at least 12 cells long).
            span = slice(d.center - 6 + m, d.center + 6 + m)
            if d.vertical:
                raw = ~L.grid.occupied[span, d.line + m]
                planned = ~L.planning_grid.occupied[span, d.line + m]
            else:
                raw = ~L.grid.occupied[d.line + m, span]
                planned = ~L.planning_grid.occupied[d.line + m, span]
            assert raw.sum() * cs == pytest.approx(DOOR_WIDTH_M)
            assert planned.sum() * cs == pytest.approx(DOOR_WIDTH_M - 2 * ROBOT_RADIUS_M)


def test_viewpoints_and_distances(layouts):
    for L in layouts:
        d = L.house.distances
        assert (d == d.T).all() and (np.diag(d) == 0).all() and (d[~np.eye(5, dtype=bool)] > 0).all()
        for k, vp in enumerate(L.viewpoints):
            assert L.planning_grid.is_free(vp) and vp in L.room_cells(k)
    L = layouts[0]
    length, _ = astar(L.planning_grid, L.viewpoints[0], L.viewpoints[3], CONNECTIVITY)
    assert length == pytest.approx(L.house.distances[0, 3])


def test_fifth_room_mix_is_kept(layouts):
    counts = collections.Counter(L.house.room_types[4] for L in layouts)
    assert set(counts) <= {"bedroom", "office", "dining"}
    assert collections.Counter(L.house.room_types[:4] for L in layouts) == {
        ("kitchen", "living", "bathroom", "bedroom"): len(layouts)
    }


def test_object_seen_from_own_room_only(layouts):
    rng = np.random.default_rng(0)
    full = 2 * np.pi
    for L in layouts[:8]:
        for room in range(5):
            assert object_candidates(L, room)
            xy = place_object(L, room, rng)
            for k in range(5):
                seen = target_visible(L.grid, (*L.viewpoint_xy(k), 0.0), xy, full)
                assert seen == (k == room)
