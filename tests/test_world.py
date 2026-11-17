"""Point 3b: table, room specs, and the uniform placement flag."""

import numpy as np
import pytest

from trustmap.cooccurrence import COOCCURRENCE, HABIT_STRENGTH, TARGETS, normalize, room_weights
from trustmap.house import CORE_ROOM_TYPES, House, draw_room_types


def test_table_has_eight_targets_with_floor():
    assert len(TARGETS) == 8
    assert sum(v == "strong" for v in HABIT_STRENGTH.values()) == 4
    for row in COOCCURRENCE.values():
        assert len(row) == 6 and min(row) >= 0.5


def test_towel_habit_worked_example(two_bedroom_house):
    # DECISIONS 3b: towel weights 1, 0.5, 8, 2, 2 sum to 13.5 -> bathroom 8/13.5.
    w = room_weights("towel", two_bedroom_house.room_types)
    assert w.tolist() == [1, 0.5, 8, 2, 2]
    assert normalize(w)[2] == pytest.approx(8 / 13.5)


def test_uniform_placement_is_all_ones(two_bedroom_house):
    assert room_weights("towel", two_bedroom_house.room_types, "uniform").tolist() == [1] * 5


def test_fifth_room_probabilities():
    rng = np.random.default_rng(0)
    fifths = [draw_room_types(rng) for _ in range(20_000)]
    assert all(t[:4] == CORE_ROOM_TYPES for t in fifths)
    share = {k: np.mean([t[4] == k for t in fifths]) for k in ("bedroom", "office", "dining")}
    assert share["bedroom"] == pytest.approx(0.50, abs=0.015)
    assert share["office"] == pytest.approx(0.25, abs=0.015)
    assert share["dining"] == pytest.approx(0.25, abs=0.015)


def test_house_rejects_bad_input():
    with pytest.raises(ValueError):
        House(room_types=("kitchen",) * 4, distances=np.zeros((4, 4)))
    with pytest.raises(ValueError):
        House(room_types=("kitchen",) * 5, distances=np.zeros((5, 5)))
