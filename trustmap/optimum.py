"""Exact optimum by value iteration (room-level MDP, Points 3a, 5 and 6).

State: (current room, set of searched rooms). The set is stored as a 5-bit
number: bit k is 1 when room k is searched. That gives 5 * 2^5 = 160 states.
Action: go search an unsearched room k (the same masked choices as the policy).

What the solver knows about the object is its belief at reset: the probability
of the object being in each room given the map's claim, the start room and the
table (3c recipe). Once some rooms are searched without success, the belief
over the remaining rooms is the reset belief restricted to them and rescaled to
sum to 1 (detection is deterministic, so "not found" rules a room out).

Bellman optimality equation, with gamma = 1 (6c):

  V(i, S) = max over k not in S of
            -d(i, k) / D                                  (cost of the trip, 6a)
            + (1 - p_find) * [ -1                  if this was the last allowed decision (6b)
                               V(k, S + {k})      otherwise ]

  where p_find = belief[k] / sum of belief over rooms not in S.
  Finding the target ends the episode with no further reward.
"""

import numpy as np

from trustmap.house import NUM_ROOMS
from trustmap.search import FAILURE_PENALTY

ALL_SEARCHED = (1 << NUM_ROOMS) - 1


def object_belief(episode, p_claim_correct):
    """Probability of the object being in each room, as known at reset.

    The claimed room gets p_claim_correct; the rest is spread over the three
    rooms that are neither claimed nor the start room, by the habit weights
    (3c step 6). The start room gets 0. With the true generator,
    p_claim_correct = episode.r (3c: P(claim correct) = r exactly).
    """
    belief = np.zeros(NUM_ROOMS)
    belief[episode.claimed_room] = p_claim_correct
    others = [k for k in range(NUM_ROOMS) if k not in (episode.claimed_room, episode.start_room)]
    habit = episode.habit[others]
    belief[others] = (1.0 - p_claim_correct) * habit / habit.sum()
    return belief


def decisions_made(searched_bits):
    """Number of decisions made so far: searched rooms minus the start room."""
    return bin(searched_bits).count("1") - 1


class OptimalSolution:
    """Value iteration result for one episode's belief and house."""

    def __init__(self, house, belief, max_decisions, tolerance=1e-12):
        self.house = house
        self.belief = np.asarray(belief, dtype=float)
        self.max_decisions = max_decisions
        self.V = np.zeros((NUM_ROOMS, ALL_SEARCHED + 1))
        self._value_iteration(tolerance)

    def _is_terminal(self, searched_bits):
        if decisions_made(searched_bits) >= self.max_decisions:
            return True
        return self._remaining_mass(searched_bits) <= 0.0

    def _remaining_mass(self, searched_bits):
        unsearched = [k for k in range(NUM_ROOMS) if not searched_bits & (1 << k)]
        return float(self.belief[unsearched].sum())

    def q_value(self, current, searched_bits, k):
        """Expected return of searching room k now and acting optimally afterwards."""
        cost = -self.house.distances[current, k] / self.house.max_distance
        p_find = self.belief[k] / self._remaining_mass(searched_bits)
        next_bits = searched_bits | (1 << k)
        if decisions_made(next_bits) == self.max_decisions:
            after_miss = -FAILURE_PENALTY
        else:
            after_miss = self.V[k, next_bits]
        return cost + (1.0 - p_find) * after_miss

    def q_values(self, current, searched_bits):
        """Q-values of all 5 rooms; searched rooms get -inf (masked)."""
        q = np.full(NUM_ROOMS, -np.inf)
        if not self._is_terminal(searched_bits):
            for k in range(NUM_ROOMS):
                if not searched_bits & (1 << k):
                    q[k] = self.q_value(current, searched_bits, k)
        return q

    def _value_iteration(self, tolerance):
        # Sweep over all states until no value changes (as in M3). The searched
        # set only grows, so this converges after a few sweeps.
        while True:
            delta = 0.0
            for current in range(NUM_ROOMS):
                for bits in range(ALL_SEARCHED + 1):
                    if not bits & (1 << current) or self._is_terminal(bits):
                        continue  # impossible state, or no decision left
                    new_value = self.q_values(current, bits).max()
                    delta = max(delta, abs(new_value - self.V[current, bits]))
                    self.V[current, bits] = new_value
            if delta < tolerance:
                break

    def start_value(self, start_room):
        """Optimal expected return of the episode, V*(start room, {start room})."""
        return float(self.V[start_room, 1 << start_room])

    def best_room(self, current, searched_bits):
        """Optimal room to search next (house room index)."""
        return int(np.argmax(self.q_values(current, searched_bits)))


def solve_episode(house, episode, max_decisions, p_claim_correct=None):
    """Optimum for one episode; p_claim_correct defaults to the true r."""
    p = episode.r if p_claim_correct is None else p_claim_correct
    return OptimalSolution(house, object_belief(episode, p), max_decisions)
