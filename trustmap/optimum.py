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

import itertools

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


# ------------------------------------------------------------------------
# Fixed search orders and the trust threshold c* (Point 9c, E2).
#
# Detection is perfect, so the only thing a failed search tells the robot is
# "not here" (8a). Every policy in one episode is therefore a fixed order of
# rooms, and its expected return is
#
#   - sum over the i-th search of  P(still not found before it) * d(prev, k_i) / D
#   - FAILURE_PENALTY * P(not in any searched room)
#
# Each of those probabilities is a sum of room probabilities, and every room
# probability is a straight line in r (r on the claimed room, (1 - r) times a
# fixed habit share on the others). So the expected return of each order is a
# straight line in r, and the optimum is the highest of these lines.


def search_orders(start_room, max_decisions):
    """Every order of `max_decisions` rooms out of the unsearched ones (24 with the cap of 3)."""
    rooms = [k for k in range(NUM_ROOMS) if k != start_room]
    return list(itertools.permutations(rooms, min(max_decisions, len(rooms))))


def order_lines(house, episode, max_decisions):
    """Each search order's expected return as a line in p = P(claim correct).

    Returns (orders, intercept, slope): the return of order j at p is
    intercept[j] + slope[j] * p.
    """
    d = house.distances / house.max_distance
    orders = search_orders(episode.start_room, max_decisions)
    ends = []
    for p in (0.0, 1.0):
        belief = object_belief(episode, p)
        values = []
        for order in orders:
            not_found = 1.0  # P(object not in any room searched so far)
            prev = episode.start_room
            total = 0.0
            for k in order:
                total -= not_found * d[prev, k]
                not_found -= belief[k]
                prev = k
            total -= FAILURE_PENALTY * max(not_found, 0.0)
            values.append(total)
        ends.append(np.array(values))
    return orders, ends[0], ends[1] - ends[0]


def order_returns(house, episode, max_decisions, p_claim_correct):
    """Expected return of every search order, for each value in p_claim_correct.

    Returns (orders, returns) with returns shaped (number of p values, number of orders).
    """
    p = np.atleast_1d(np.asarray(p_claim_correct, dtype=float))
    orders, intercept, slope = order_lines(house, episode, max_decisions)
    return orders, intercept[None, :] + p[:, None] * slope[None, :]


def _follow(intercept, slope, claim_first, p, tolerance=1e-12):
    returns = intercept[None, :] + np.atleast_1d(p)[:, None] * slope[None, :]
    return returns[:, claim_first].max(axis=1) >= returns[:, ~claim_first].max(axis=1) - tolerance


def follows_claim(house, episode, max_decisions, p_claim_correct):
    """True where the optimum's first search is the claimed room (ties count as following)."""
    orders, intercept, slope = order_lines(house, episode, max_decisions)
    claim_first = np.array([o[0] == episode.claimed_room for o in orders])
    return _follow(intercept, slope, claim_first, np.asarray(p_claim_correct, dtype=float))


def confidence_threshold(house, episode, max_decisions, grid_points=2001):
    """c* of one episode: the reliability at which the optimum starts searching the claimed room first.

    Everything except r is kept as drawn (house, start room, claimed room,
    target). Returns None when the optimum never switches: it follows the
    claim for every r, for none, or only at r = 1 as a tie. If it switches more than once, c* is the
    last switch, above which it always follows. The switch is first located on
    a grid of r values and then narrowed down by bisection to about 1e-12.
    """
    orders, intercept, slope = order_lines(house, episode, max_decisions)
    claim_first = np.array([o[0] == episode.claimed_room for o in orders])
    r = np.linspace(0.0, 1.0, grid_points)
    follow = _follow(intercept, slope, claim_first, r)
    if not follow[-1] or follow.all():
        return None
    i = int(np.flatnonzero(~follow)[-1])  # last grid r where it does not follow
    lo, hi = float(r[i]), float(r[i + 1])
    for _ in range(40):  # bisection: halve the interval that contains the switch
        mid = (lo + hi) / 2.0
        if _follow(intercept, slope, claim_first, mid)[0]:
            hi = mid
        else:
            lo = mid
    if hi >= 1.0 - 1e-9:
        # The claim only ties at r = 1: some room lies exactly on a shortest path to
        # the claimed room, so searching it first is free and strictly better for
        # every r < 1. The optimum never really switches, so there is no c*.
        return None
    return hi
