"""Points 5 and 6: masking, episode end, reward."""

import numpy as np
import pytest

from conftest import BATHROOM, BEDROOM_A, BEDROOM_B, KITCHEN, LIVING
from tests.test_observation import make_episode
from trustmap.metrics import episode_metrics
from trustmap.search import RoomSearch


def row_of(episode, room):
    return int(np.where(episode.row_order == room)[0][0])


def test_start_room_masked_then_each_searched_room(two_bedroom_house):
    ep = make_episode([2, 0, 4, 1, 3])  # object in bathroom, start kitchen
    s = RoomSearch(two_bedroom_house, ep, max_decisions=3)
    assert s.action_mask().tolist() == [True, False, True, True, True]
    s.step(row_of(ep, LIVING))
    assert not s.action_mask()[row_of(ep, LIVING)]
    with pytest.raises(ValueError):
        s.step(row_of(ep, KITCHEN))


def test_success_on_detection_with_scaled_reward(two_bedroom_house):
    ep = make_episode([0, 1, 2, 3, 4])
    s = RoomSearch(two_bedroom_house, ep, max_decisions=3)
    r1 = s.step(LIVING)     # kitchen -> living: 4 m, house max 9 m
    r2 = s.step(BATHROOM)   # living -> bathroom: 5 m, object found
    assert r1 == pytest.approx(-4 / 9) and r2 == pytest.approx(-5 / 9)
    assert s.done and s.success and s.distance_m == 9.0


def test_failure_after_three_misses(two_bedroom_house):
    ep = make_episode([0, 1, 2, 3, 4])  # object in bathroom
    s = RoomSearch(two_bedroom_house, ep, max_decisions=3)
    rewards = [s.step(LIVING), s.step(BEDROOM_A), s.step(BEDROOM_B)]
    assert s.done and not s.success
    assert rewards[2] == pytest.approx(-2 / 9 - 1.0)  # last trip plus P = 1
    with pytest.raises(RuntimeError):
        s.step(BATHROOM)


def test_metrics(two_bedroom_house):
    ep = make_episode([0, 1, 2, 3, 4])  # start kitchen, object bathroom: shortest 7 m
    s = RoomSearch(two_bedroom_house, ep, max_decisions=3)
    s.step(LIVING)
    s.step(BATHROOM)  # 4 + 5 = 9 m taken
    m = episode_metrics(s)
    assert m.success and not m.success_first_decision
    assert m.spl == pytest.approx(7 / 9)

    s = RoomSearch(two_bedroom_house, ep, max_decisions=3)
    s.step(BATHROOM)
    m = episode_metrics(s)
    assert m.success_first_decision and m.spl == pytest.approx(1.0)

    s = RoomSearch(two_bedroom_house, ep, max_decisions=3)
    for room in (LIVING, BEDROOM_A, BEDROOM_B):
        s.step(room)
    assert episode_metrics(s).spl == 0.0


def test_cap_of_four_always_succeeds(two_bedroom_house):
    ep = make_episode([0, 1, 2, 3, 4])
    s = RoomSearch(two_bedroom_house, ep, max_decisions=4)
    for room in (LIVING, BEDROOM_A, BEDROOM_B, BATHROOM):
        s.step(room)
    assert s.success
