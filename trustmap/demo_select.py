"""Pick the Gazebo demo episodes by a fixed rule (DECISIONS "Demo material"; demo only).

    python -m trustmap.demo_select        # -> results/demo/selection.json

Candidates: the 25 E1 test episodes (9b, seed 200,000) of each demo house
200000, 200001, 200002 and 200003 at rho = 0.4, played in 2D by the exported
PPO-Conf policy (gazebo/policies/ppo_conf.npz, the median-validation seed).
Decisions in Gazebo are identical to 2D (same observation code), so the 2D
outcome is the Gazebo decision outcome.

Rule (stated in results/demo/README.md):
  successes: houses in order; in each house, the first 2 successful episodes
             (by episode index) whose target is not already selected; if a
             house has no such episode left, the first successes with any target;
  failures:  the first 2 failed episodes in scan order (house order, then
             episode index) at rho = 0.4; if fewer than 2 exist there, the scan is
             repeated at rho = 0.6 and then 0.8, and those episodes are labelled
             with their rho.
That gives about 10 episodes: 8 successes covering every target, 2 failures.
These are hand-picked illustrations, not part of the Gazebo result rows.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from trustmap.episode import DEFAULT_MAX_DECISIONS, EpisodeConfig
from trustmap.evaluation import make_test_episodes, play
from trustmap.metrics import episode_metrics
from trustmap.optimum import solve_episode
from trustmap.pool import load_pool

REPO = Path(__file__).resolve().parent.parent
DEMO_HOUSES = (200_000, 200_001, 200_002, 200_003)
POLICY = REPO / "gazebo" / "policies" / "ppo_conf.npz"
PER_HOUSE = 25            # 9b: test episodes per house
TEST_SEED = 200_000       # 9b
SUCCESSES_PER_HOUSE = 2
FAILURES = 2


def house_episodes(seed, rho, houses=None):
    """The 25 E1 test episodes of test house `seed` at this rho, as (index, house, episode)."""
    houses = houses or load_pool("test")
    index = [h.seed for h in houses].index(seed)
    all_eps = make_test_episodes(houses, EpisodeConfig(rho=rho, kappa=1.0), PER_HOUSE, TEST_SEED)
    return [(i, h, ep) for i, (h, ep) in enumerate(all_eps[index * PER_HOUSE:(index + 1) * PER_HOUSE])]


def candidates(rho, houses):
    from trustmap.policy_export import NumpyPolicy

    policy = NumpyPolicy(str(POLICY), name="ppo_conf")
    out = []
    for seed in DEMO_HOUSES:
        for i, house, ep in house_episodes(seed, rho, houses):
            search = play(policy, house, ep, DEFAULT_MAX_DECISIONS, np.random.default_rng([seed, i]))
            m = episode_metrics(search, solve_episode(house, ep, DEFAULT_MAX_DECISIONS))
            out.append({"house": seed, "episode": i, "rho": rho, "target": ep.target,
                        "success": bool(m.success), "decisions": m.decisions,
                        "regret_m": round(float(m.regret_m), 3), "planned_m": round(m.distance_m, 2),
                        "c": round(float(ep.c), 3), "claim_correct": bool(ep.claim_correct)})
    return out


def select(houses=None):
    houses = houses or load_pool("test")
    cands = candidates(0.4, houses)
    chosen, targets = [], set()
    for seed in DEMO_HOUSES:
        wins = [c for c in cands if c["house"] == seed and c["success"]]
        picked = []
        for c in wins:
            if len(picked) < SUCCESSES_PER_HOUSE and c["target"] not in targets:
                picked.append(c)
                targets.add(c["target"])
        for c in wins:  # fallback: any target
            if len(picked) < SUCCESSES_PER_HOUSE and c not in picked:
                picked.append(c)
        chosen += picked
    failures = [c for c in cands if not c["success"]][:FAILURES]
    for rho in (0.6, 0.8):
        if len(failures) >= FAILURES:
            break
        failures += [c for c in candidates(rho, houses) if not c["success"]][:FAILURES - len(failures)]
    chosen += failures
    n_fail_04 = sum(1 for c in cands if not c["success"])
    return chosen, {"candidates_at_rho_0.4": len(cands), "failures_at_rho_0.4": n_fail_04}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(REPO / "results" / "demo" / "selection.json"))
    a = p.parse_args()
    chosen, info = select()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"rule": __doc__.split("Rule (stated in results/demo/README.md):")[1]
                                       .split("These are")[0].strip(), **info, "episodes": chosen}, indent=2))
    for c in chosen:
        print(c)
    print(info)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
