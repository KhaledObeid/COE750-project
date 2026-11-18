"""E0: tabular Q-learning reaches the exact optimum (Point 8c).

    python -m trustmap.e0
    python -m trustmap.e0 --figures-only

For a fixed set of episode configurations (house, target, claim, r, start
room), drawn from the test houses at rho 0.2, 0.4, 0.6 and 0.8 (5 each), tabular
Q-learning (M4-2) is trained on the room-level table, state = (current room,
searched set), and compared during training with the value-iteration
Q-values. Recorded after 500 ... 50,000 training episodes:

  start_q_error      |Q(start, best room) - Q*(start, best room)|, in house-widths
  start_agreement    the greedy first room equals the optimum's first room
  path_agreement     share of the states on the optimum's own search path (the
                     states it passes through while the object is not found)
                     where the greedy room equals the optimum's
  state_agreement    the same over all decision states before the cap; lower,
                     because states reached only after an unusual choice are
                     visited only through epsilon exploration and are updated
                     rarely, and some hold near-ties between two rooms

This links the project to the course: when the state is a small table,
tabular RL reaches the optimum; what PPO adds is generalization across houses
and confidence values. Outputs: results/e0/curve.csv, results/figures/e0_qlearning.
"""

import argparse
import multiprocessing

import numpy as np

from trustmap.episode import DEFAULT_MAX_DECISIONS, EpisodeConfig
from trustmap.evaluation import make_test_episodes
from trustmap.experiments import RESULTS_DIR, TEST_SEED, save_figure, t_interval, write_csv
from trustmap.house import NUM_ROOMS
from trustmap.optimum import ALL_SEARCHED, decisions_made, solve_episode
from trustmap.pool import load_pool
from trustmap.qlearning import q_learning

RHOS = (0.2, 0.4, 0.6, 0.8)
CONFIGS_PER_RHO = 5
CHECKPOINTS = (500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000)


def decision_states(start_room, max_decisions):
    """(current room, searched bits) of every state where a decision is still made."""
    out = []
    for bits in range(ALL_SEARCHED + 1):
        if not bits & (1 << start_room) or decisions_made(bits) >= max_decisions:
            continue
        for current in range(NUM_ROOMS):
            if bits & (1 << current) and (current == start_room or bits != (1 << start_room)):
                out.append((current, bits))
    return out


def run_config(job):
    index, house, ep = job
    cap = DEFAULT_MAX_DECISIONS
    sol = solve_episode(house, ep, cap)
    start_bits = 1 << ep.start_room
    best = sol.best_room(ep.start_room, start_bits)
    states = [s for s in decision_states(ep.start_room, cap) if np.isfinite(sol.q_values(*s)).any()]
    path, current, bits = [], ep.start_room, start_bits  # the optimum's own search path
    while decisions_made(bits) < cap and np.isfinite(sol.q_values(current, bits)).any():
        path.append((current, bits))
        current = sol.best_room(current, bits)
        bits |= 1 << current
    rows = []

    def record(n, Q):
        if n not in CHECKPOINTS:
            return
        q_star = sol.q_values(ep.start_room, start_bits)
        allowed = np.isfinite(q_star)
        greedy = int(np.argmax(np.where(allowed, Q[ep.start_room, start_bits], -np.inf)))
        def agrees(current, bits):
            qs = sol.q_values(current, bits)
            ok = np.isfinite(qs)
            return int(np.argmax(np.where(ok, Q[current, bits], -np.inf))) == int(np.argmax(qs))
        rows.append({"config": index, "episodes": n,
                     "start_q_error": abs(Q[ep.start_room, start_bits, best] - q_star[best]),
                     "start_agreement": float(greedy == best),
                     "path_agreement": float(np.mean([agrees(*st) for st in path])),
                     "state_agreement": float(np.mean([agrees(*st) for st in states]))})

    q_learning(house, ep, cap, num_episodes=max(CHECKPOINTS), seed=index, on_episode=record)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figures-only", action="store_true")
    a = p.parse_args()
    path = RESULTS_DIR / "e0" / "per_config.csv"
    if not a.figures_only:
        houses = load_pool("test")
        jobs = []
        for rho in RHOS:
            eps = make_test_episodes(houses, EpisodeConfig(rho=rho, kappa=1.0), 25, TEST_SEED)
            for i in range(CONFIGS_PER_RHO):  # spread over the 5,000 episodes of this rho
                jobs.append((len(jobs), *eps[i * 997 % len(eps)]))
        print(f"E0: {len(jobs)} configurations x {max(CHECKPOINTS)} Q-learning episodes")
        with multiprocessing.Pool() as pool:
            rows = [r for rs in pool.map(run_config, jobs) for r in rs]
        write_csv(path, rows)

    import csv

    with open(path) as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    curve = []
    for n in CHECKPOINTS:
        sel = [r for r in rows if r["episodes"] == n]
        row = {"episodes": n}
        for k in ("start_q_error", "start_agreement", "path_agreement", "state_agreement"):
            mean, half = t_interval([r[k] for r in sel])
            row[k], row[f"{k}_ci95"] = round(mean, 4), round(half, 4)
        curve.append(row)
    write_csv(RESULTS_DIR / "e0" / "curve.csv", curve)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["episodes"] for r in curve]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].errorbar(x, [r["start_q_error"] for r in curve], yerr=[r["start_q_error_ci95"] for r in curve],
                     marker="o", capsize=3)
    axes[0].set_ylabel("|Q - Q*| at the start (house-widths)")
    for k, label in (("start_agreement", "first room"), ("path_agreement", "states on the optimum's path"),
                     ("state_agreement", "all decision states")):
        axes[1].errorbar(x, [r[k] for r in curve], yerr=[r[f"{k}_ci95"] for r in curve], marker="o",
                         capsize=3, label=label)
    axes[1].set_ylabel("greedy choice = optimum")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Q-learning training episodes")
        ax.grid(alpha=0.3)
    fig.suptitle("E0: tabular Q-learning against value iteration", fontsize=10)
    save_figure(fig, "e0_qlearning")
    plt.close(fig)


if __name__ == "__main__":
    main()
