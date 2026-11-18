"""The four baselines of Point 8d.

Each baseline is a small object with `choose(search, rng) -> row`: given the
episode being played (a `RoomSearch`), it returns the observation row to
search next. All of them obey the 5a mask (only unsearched rooms) and the
3-decision cap (which `RoomSearch` enforces). The PPO agents are wrapped in
the same interface by the evaluation code, so every method is played by the
same loop.

  Random              uniformly among unsearched rooms (no strategy)
  NearestUnexplored   nearest unsearched room by planned distance (distance only)
  GreedyMapFollower   the claimed room first, then nearest unsearched (blind trust)
  BayesNominal        the exact optimum computed with the displayed c taken as r
                      (correct use of a confidence that may lie; equals the
                      optimum when kappa = 1)

The proposal's Bayes-oracle (true r) is the optimum itself, the regret-0
reference line, so it is not a separate baseline (8d).
"""

import numpy as np

from trustmap.house import NUM_ROOMS
from trustmap.optimum import solve_episode


def unsearched_rooms(search):
    return [k for k in range(NUM_ROOMS) if not search.searched[k]]


def nearest_unsearched_room(search):
    """Unsearched room with the shortest planned distance from the current room.

    Exact ties are broken by the episode's shuffled row order (the room shown
    higher up wins), which is random per episode, so no room type is favored.
    """
    rooms = unsearched_rooms(search)
    dist = search.house.distances[search.current_room]
    return min(rooms, key=lambda k: (dist[k], search.row_of_room(k)))


class Random:
    name = "Random"

    def choose(self, search, rng):
        return search.row_of_room(int(rng.choice(unsearched_rooms(search))))


class NearestUnexplored:
    name = "Nearest-unexplored"

    def choose(self, search, rng):
        return search.row_of_room(nearest_unsearched_room(search))


class GreedyMapFollower:
    name = "Greedy map follower"

    def choose(self, search, rng):
        claimed = search.episode.claimed_room
        if not search.searched[claimed]:
            return search.row_of_room(claimed)
        return search.row_of_room(nearest_unsearched_room(search))


class BayesNominal:
    """Plans exactly, but believes the displayed confidence c is the true reliability."""

    name = "Bayes-nominal"

    def __init__(self):
        self._episode = None
        self._solution = None

    def choose(self, search, rng):
        if self._episode is not search.episode:  # new episode: solve once, reuse after
            self._episode = search.episode
            self._solution = solve_episode(
                search.house, search.episode, search.max_decisions, p_claim_correct=search.episode.c
            )
        q = self._solution.q_values(search.current_room, search.searched_bits)
        if np.isinf(q).all():
            # Only possible when c rounds to exactly 1 but the claim was wrong:
            # the nominal belief has no probability left for any room. Fall
            # back to the nearest unsearched room so the episode can continue.
            return search.row_of_room(nearest_unsearched_room(search))
        return search.row_of_room(int(np.argmax(q)))


ALL_BASELINES = (Random, NearestUnexplored, GreedyMapFollower, BayesNominal)
