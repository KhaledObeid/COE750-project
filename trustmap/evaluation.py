"""Play policies on a fixed list of episodes and collect metrics.

Paired evaluation (Agreements): episodes are drawn once from a seed, before
any policy plays, so every method faces the identical houses, targets, claims,
confidences, start rooms and object rooms. All randomness of an episode is at
reset, so the policy's own choices can never change which episode comes next.

A policy is any object with `choose(search, rng) -> row` (see baselines.py;
PPO agents are wrapped in ppo.py).

Which rho values, how many episodes and which seed the experiments use are
Point 9 decisions; this module only takes them as arguments.
"""

import dataclasses
from dataclasses import asdict

import numpy as np

from trustmap.episode import generate_episode
from trustmap.metrics import episode_metrics
from trustmap.optimum import solve_episode
from trustmap.search import RoomSearch


def make_episodes(houses, config, num_episodes, seed, rho_range=None):
    """Draw `num_episodes` (house, episode) pairs; the same seed always gives the same list.

    With rho_range = (low, high), rho is drawn uniformly per episode instead
    of using config.rho (as the training environment does).
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num_episodes):
        cfg = config if rho_range is None else dataclasses.replace(config, rho=float(rng.uniform(*rho_range)))
        house = houses[rng.integers(len(houses))]
        out.append((house, generate_episode(house, cfg, rng)))
    return out


def make_test_episodes(houses, config, per_house, seed):
    """Test set of Point 9b: exactly `per_house` episodes in every house, in house order.

    Each episode has its own target, claim, confidence and start room. The
    same seed is used for every method and every rho (paired evaluation).
    """
    rng = np.random.default_rng(seed)
    return [(house, generate_episode(house, config, rng)) for house in houses for _ in range(per_house)]


def solve_all(episodes, max_decisions):
    """Optimum (true r, 8a) for every episode; about 5 ms each, so do it once and reuse."""
    return [solve_episode(house, ep, max_decisions) for house, ep in episodes]


def play(policy, house, episode, max_decisions, rng):
    search = RoomSearch(house, episode, max_decisions)
    while not search.done:
        search.step(policy.choose(search, rng))
    return search


def evaluate(policy, episodes, max_decisions, solutions, seed=0):
    """Metrics (with regret) of `policy` on every episode, in order."""
    rng = np.random.default_rng(seed)  # only the Random baseline uses it
    return [
        episode_metrics(play(policy, house, ep, max_decisions, rng), sol)
        for (house, ep), sol in zip(episodes, solutions)
    ]


def summarize(metrics):
    """Averages over episodes: success rate, SPL, first-decision success, regret and distance in meters."""
    rows = [asdict(m) for m in metrics]
    return {
        "episodes": len(rows),
        "success": float(np.mean([r["success"] for r in rows])),
        "spl": float(np.mean([r["spl"] for r in rows])),
        "success_first_decision": float(np.mean([r["success_first_decision"] for r in rows])),
        "regret_m": float(np.mean([r["regret_m"] for r in rows])),
        "distance_m": float(np.mean([r["distance_m"] for r in rows])),
    }
