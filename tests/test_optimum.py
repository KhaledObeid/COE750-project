"""Exact optimum: compared against trying every room order with RoomSearch."""

import dataclasses
import itertools

import numpy as np
import pytest

from trustmap.episode import EpisodeConfig, generate_episode
from trustmap.optimum import object_belief, solve_episode
from trustmap.search import RoomSearch


def play_order(house, episode, order, max_decisions):
    """Return of searching rooms in a fixed order until the episode ends."""
    s = RoomSearch(house, episode, max_decisions)
    total = 0.0
    for room in order:
        if s.done:
            break
        total += s.step(int(np.where(episode.row_order == room)[0][0]))
    return total


def expected_return_of_order(house, episode, order, max_decisions, belief):
    """Average the played return over where the object could be."""
    total = 0.0
    for room, p in enumerate(belief):
        if p > 0:
            ep = dataclasses.replace(episode, object_room=room)
            total += p * play_order(house, ep, order, max_decisions)
    return total


@pytest.mark.parametrize("max_decisions", [2, 3, 4])
def test_value_iteration_matches_best_fixed_order(two_bedroom_house, max_decisions):
    # Detection is perfect, so the only thing learned along the way is "not
    # here"; the best policy is therefore the best fixed order of rooms.
    rng = np.random.default_rng(3)
    cfg = EpisodeConfig(rho=0.4, kappa=1.0)
    for _ in range(30):
        ep = generate_episode(two_bedroom_house, cfg, rng)
        belief = object_belief(ep, ep.r)
        sol = solve_episode(two_bedroom_house, ep, max_decisions)
        unsearched = [k for k in range(5) if k != ep.start_room]
        best = max(expected_return_of_order(two_bedroom_house, ep, order, max_decisions, belief)
                   for order in itertools.permutations(unsearched))
        assert sol.start_value(ep.start_room) == pytest.approx(best)


def test_belief_sums_to_one_and_excludes_start(two_bedroom_house):
    rng = np.random.default_rng(4)
    for _ in range(50):
        ep = generate_episode(two_bedroom_house, EpisodeConfig(rho=0.6, kappa=1.0), rng)
        b = object_belief(ep, ep.r)
        assert b.sum() == pytest.approx(1.0)
        assert b[ep.start_room] == 0.0 and b[ep.claimed_room] == pytest.approx(ep.r)


def test_perfect_map_goes_straight_to_claim(two_bedroom_house):
    rng = np.random.default_rng(5)
    ep = generate_episode(two_bedroom_house, EpisodeConfig(rho=0.0, kappa=1.0), rng)
    sol = solve_episode(two_bedroom_house, ep, 3)
    start_bits = 1 << ep.start_room
    assert sol.best_room(ep.start_room, start_bits) == ep.claimed_room
    d = two_bedroom_house.distances[ep.start_room, ep.claimed_room] / two_bedroom_house.max_distance
    assert sol.start_value(ep.start_room) == pytest.approx(-d)
