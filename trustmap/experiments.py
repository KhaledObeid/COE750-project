"""Shared machinery for the experiments E1-E3 (Point 9).

Every experiment follows the same recipe:
  1. draw the Point 9b test set for each rho: 25 episodes in each of the 200
     test houses (5,000 per rho), seed 200,000, identical for every method;
  2. solve the exact optimum of every episode once (8a), shared by all methods;
  3. let every method play every episode and keep one record per episode
     (regret, success, SPL, first-decision success, distance, and whether the
     first search was the claimed room);
  4. save the records to results/<experiment>/, so tables and figures can be
     redrawn at any time without evaluating again (Point 11a, 11c).

Reported numbers are the mean over training seeds with a 95% confidence
interval from the seed results (9b): a t-interval, mean +- t * sd / sqrt(n),
the range where the true average most likely lies given how much the seeds
disagree. Baselines have no training seed, so they get a single value.

The rho values of each experiment are evaluated in parallel processes.
"""

import csv
import multiprocessing
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trustmap.baselines import BayesNominal, GreedyMapFollower, NearestUnexplored, Random
from trustmap.cooccurrence import HABIT_STRENGTH, TARGETS
from trustmap.episode import DEFAULT_LAMBDA, DEFAULT_MAX_DECISIONS, EpisodeConfig
from trustmap.evaluation import make_test_episodes, play, solve_all
from trustmap.metrics import episode_metrics
from trustmap.observation import Variant
from trustmap.optimum import confidence_threshold
from trustmap.pool import load_pool

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RUNS_DIR = REPO / "runs"

TEST_RHOS = (0.0, 0.2, 0.4, 0.6, 0.8)  # 9a
EPISODES_PER_HOUSE = 25                # 9b: 25 x 200 test houses = 5,000 per rho
TEST_SEED = 200_000                    # 9b
RECORD_FIELDS = ("regret_m", "success", "spl", "success_first_decision", "distance_m", "followed")


@dataclass(frozen=True)
class Condition:
    """The dials of one test condition (kappa, lambda, placement); rho varies inside it."""

    name: str
    kappa: float = 1.0
    lam: float = DEFAULT_LAMBDA
    placement: str = "table"

    def config(self, rho):
        return EpisodeConfig(rho=rho, kappa=self.kappa, lam=self.lam, placement=self.placement,
                             max_decisions=DEFAULT_MAX_DECISIONS)


@dataclass(frozen=True)
class Method:
    """One method to evaluate: a baseline (no seed) or a trained PPO run (one seed)."""

    key: str                  # e.g. "ppo_conf", "follower"
    label: str                # name in tables and figures
    seed: int | None = None   # training seed of a PPO run
    run_dir: Path | None = None
    variant: Variant | None = None


class Optimum:
    """Plays the exact optimum (true r, 8a). It is the regret-0 reference line, shown for its
    success, SPL and first-decision success (Q8: first-decision success is always shown next
    to the optimum's own value)."""

    name = "Optimum"

    def __init__(self, solutions_by_episode):
        self.solutions = solutions_by_episode

    def choose(self, search, rng):
        sol = self.solutions[id(search.episode)]
        return search.row_of_room(sol.best_room(search.current_room, search.searched_bits))


BASELINES = (
    Method("random", "Random"),
    Method("nearest", "Nearest-unexplored"),
    Method("follower", "Greedy map follower"),
    Method("bayes_nominal", "Bayes-nominal"),
)
OPTIMUM = Method("optimum", "Optimum")


def ppo_methods(key, label, variant_key, seeds, kappa=1.0, lam=DEFAULT_LAMBDA, placement="table"):
    """Finished PPO runs under runs/ (evaluated with their best checkpoint, 7d).

    A run counts only once training has finished (final_model.zip exists), so a
    run still in progress is never evaluated with an intermediate checkpoint.
    Missing or unfinished runs are reported and skipped.
    """
    from trustmap.train import VARIANTS, run_name

    out = []
    for s in seeds:
        run_dir = RUNS_DIR / run_name(variant_key, s, kappa, lam, placement)
        if (run_dir / "final_model.zip").exists():
            out.append(Method(key, label, s, run_dir, VARIANTS[variant_key]))
        else:
            print(f"  missing or unfinished run {run_dir.name}, skipped")
    return out


# --------------------------------------------------------------- evaluation ---

def _policy(method, solutions_by_episode):
    if method.key == "random":
        return Random()
    if method.key == "nearest":
        return NearestUnexplored()
    if method.key == "follower":
        return GreedyMapFollower()
    if method.key == "bayes_nominal":
        return BayesNominal()
    if method.key == "optimum":
        return Optimum(solutions_by_episode)
    import torch
    from sb3_contrib import MaskablePPO

    from trustmap.ppo import PPOPolicy

    torch.set_num_threads(1)
    return PPOPolicy(MaskablePPO.load(method.run_dir / "best_model.zip", device="cpu"), method.variant)


def episode_table(episodes, max_decisions, with_c_star):
    """Facts about each test episode, shared by all methods (used to split results)."""
    targets = np.array([TARGETS.index(ep.target) for _, ep in episodes])
    table = {
        "house_seed": np.array([h.seed for h, _ in episodes]),
        "target": targets,
        "weak_habit": np.array([HABIT_STRENGTH[TARGETS[t]] == "weak" for t in targets]),
        "c": np.array([ep.c for _, ep in episodes]),
        "r": np.array([ep.r for _, ep in episodes]),
        "claim_distance": np.array([h.distances[ep.start_room, ep.claimed_room] / h.max_distance
                                    for h, ep in episodes]),
        "house_max_distance_m": np.array([h.max_distance for h, _ in episodes]),
    }
    if with_c_star:
        c_star = [confidence_threshold(h, ep, max_decisions) for h, ep in episodes]
        table["c_star"] = np.array([np.nan if x is None else x for x in c_star])
    return table


def play_records(method, episodes, solutions, seed):
    """One record per episode for one method."""
    by_episode = {id(ep): sol for (_, ep), sol in zip(episodes, solutions)}
    policy = _policy(method, by_episode)
    rng = np.random.default_rng(seed)  # only Random uses it
    rec = {f: [] for f in RECORD_FIELDS}
    for (house, ep), sol in zip(episodes, solutions):
        search = play(policy, house, ep, DEFAULT_MAX_DECISIONS, rng)
        m = episode_metrics(search, sol)
        rec["regret_m"].append(m.regret_m)
        rec["success"].append(m.success)
        rec["spl"].append(m.spl)
        rec["success_first_decision"].append(m.success_first_decision)
        rec["distance_m"].append(m.distance_m)
        rec["followed"].append(search.history[0][2] == ep.claimed_room)
    return {f: np.array(v, dtype=float) for f, v in rec.items()}


def _evaluate_rho(job):
    condition, rho, methods, out_path, with_c_star, per_house = job
    cfg = condition.config(rho)
    episodes = make_test_episodes(load_pool("test"), cfg, per_house, TEST_SEED)
    solutions = solve_all(episodes, cfg.max_decisions)
    data = {f"episode/{k}": v for k, v in episode_table(episodes, cfg.max_decisions, with_c_star).items()}
    for m in methods:
        for f, v in play_records(m, episodes, solutions, TEST_SEED).items():
            data[f"{m.key}/{m.seed}/{f}"] = v
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    return out_path


def _show(path):
    """Path relative to the repository when possible, for short log lines."""
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


def records_path(experiment, condition, rho):
    return RESULTS_DIR / experiment / f"{condition.name}_rho{rho:g}.npz"


def evaluate_condition(experiment, condition, rhos, methods, with_c_star=False, workers=None,
                       per_house=EPISODES_PER_HOUSE):
    """Evaluate all methods at every rho of one condition and save the records.

    per_house is 25 for every reported result (9b); tests use fewer.
    """
    jobs = [(condition, rho, methods, records_path(experiment, condition, rho), with_c_star, per_house)
            for rho in rhos]
    workers = workers or min(len(jobs), os.cpu_count() or 1)
    print(f"{experiment}/{condition.name}: {len(methods)} method runs x {len(rhos)} rho values, "
          f"{workers} processes")
    with multiprocessing.Pool(workers) as pool:
        for path in pool.imap_unordered(_evaluate_rho, jobs):
            print(f"  saved {_show(path)}")


# ----------------------------------------------------------- loading/stats ---

@dataclass
class Records:
    """Saved records of one condition at one rho."""

    episode: dict
    runs: dict = field(default_factory=dict)  # (method key, seed) -> {field: array}

    def seeds(self, key):
        return sorted(s for k, s in self.runs if k == key)

    def per_seed(self, key, fld, mask=None):
        """Mean of one field for every seed of a method (one value for a baseline)."""
        out = []
        for s in self.seeds(key):
            v = self.runs[(key, s)][fld]
            out.append(float(v[mask].mean() if mask is not None else v.mean()))
        return np.array(out)


def load_records(experiment, condition, rho):
    data = np.load(records_path(experiment, condition, rho))
    rec = Records({k.split("/", 1)[1]: data[k] for k in data.files if k.startswith("episode/")})
    for k in data.files:
        if k.startswith("episode/"):
            continue
        key, seed, fld = k.split("/")
        seed = None if seed == "None" else int(seed)
        rec.runs.setdefault((key, seed), {})[fld] = data[k]
    return rec


# Two-sided 95% t quantiles for n - 1 degrees of freedom (from a standard t table).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 19: 2.093, 24: 2.064, 29: 2.045}


def t_interval(values):
    """(mean, half-width) of a 95% t-interval over seed results; half-width 0 for a single value."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(values[0]), 0.0
    df = n - 1
    if df > 29:
        t = 1.96  # the t quantile is close to the normal one from here on
    else:
        t = _T95[max(k for k in _T95 if k <= df)]  # exact for 1-15 degrees of freedom
    return float(values.mean()), float(t * values.std(ddof=1) / np.sqrt(n))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {_show(path)}")


def save_figure(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    print(f"  wrote {_show(FIGURES_DIR / name)}.pdf/.png")
