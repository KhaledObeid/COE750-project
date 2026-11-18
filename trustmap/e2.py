"""E2: trust analysis against each episode's own c* (Point 9c), for H2.

    python -m trustmap.e2        (needs the E1 main records: run python -m trustmap.e1 first)

For every E1 test episode at kappa 1 (so c = r), c* is the confidence at which
the optimum's first choice switches to the claimed room, with everything else
in the episode kept as drawn (computed in E1, see optimum.confidence_threshold).
The main figure shows, for PPO-Conf, PPO-Binary, the greedy follower and the
optimum, the fraction of episodes whose first search is the claimed room,
against c - c*. A policy that tracks c* jumps from rarely to almost always
following at 0, whatever the distance (H2). Episodes without a c* (the
optimum never switches) are left out and counted.

Panels: (a) follow rate against c - c* for all four methods; (b) PPO-Conf
alone, split into episodes whose claimed room is near or far (below or above
the median scaled distance), to show the jump does not depend on distance;
(c) the simpler trust curve by confidence bin, from the same data.
Outputs: results/e2/follow.csv, results/e2/summary.csv, results/figures/e2_trust.
"""

import argparse

import numpy as np

from trustmap.e1 import CONDITIONS
from trustmap.experiments import RESULTS_DIR, TEST_RHOS, load_records, save_figure, t_interval, write_csv

METHODS = (("ppo_conf", "PPO-Conf"), ("ppo_binary", "PPO-Binary"), ("follower", "Greedy map follower"),
           ("optimum", "Optimum"))
GAP_EDGES = np.round(np.arange(-1.0, 1.0001, 0.05), 3)   # bins of c - c* (c and c* both lie in [0, 1])
CONF_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0001])  # bins of c for panel (c)
MIN_EPISODES = 30                                         # bins with fewer episodes are not drawn


def pooled():
    """Episode facts and per-seed 'followed' arrays, pooled over the E1 rho values."""
    recs = [load_records("e1", CONDITIONS["main"], rho) for rho in TEST_RHOS]
    ep = {k: np.concatenate([r.episode[k] for r in recs]) for k in ("c", "c_star", "claim_distance")}
    followed = {}
    for key, _ in METHODS:
        seeds = recs[0].seeds(key)
        followed[key] = [np.concatenate([r.runs[(key, s)]["followed"] for r in recs]) for s in seeds]
    return ep, followed


def curve(values, edges, followed_runs, mask):
    """Follow rate per bin: mean over seeds with 95% interval, plus the episode count."""
    out = []
    which = np.digitize(values, edges) - 1
    for b in range(len(edges) - 1):
        sel = mask & (which == b)
        n = int(sel.sum())
        if n < MIN_EPISODES:
            continue
        mean, half = t_interval([f[sel].mean() for f in followed_runs])
        out.append(((edges[b] + edges[b + 1]) / 2, mean, half, n))
    return out


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    ep, followed = pooled()
    has_c_star = ~np.isnan(ep["c_star"])
    gap = ep["c"] - ep["c_star"]
    near = ep["claim_distance"] <= np.median(ep["claim_distance"])

    rows, curves = [], {}
    for key, label in METHODS:
        if not followed[key]:
            print(f"  no records for {label}, skipped")
            continue
        curves[key] = curve(gap, GAP_EDGES, followed[key], has_c_star)
        rows += [{"panel": "a", "method": label, "bin_center": round(x, 3), "follow_rate": round(m, 4),
                  "ci95": round(h, 4), "episodes": n} for x, m, h, n in curves[key]]
        for x, m, h, n in curve(ep["c"], CONF_EDGES, followed[key], np.ones_like(has_c_star)):
            rows.append({"panel": "c", "method": label, "bin_center": round(x, 3), "follow_rate": round(m, 4),
                         "ci95": round(h, 4), "episodes": n})
    split = {}
    if followed.get("ppo_conf"):
        for name, mask in (("near", near), ("far", ~near)):
            split[name] = curve(gap, GAP_EDGES, followed["ppo_conf"], has_c_star & mask)
            rows += [{"panel": f"b-{name}", "method": "PPO-Conf", "bin_center": round(x, 3),
                      "follow_rate": round(m, 4), "ci95": round(h, 4), "episodes": n}
                     for x, m, h, n in split[name]]
    write_csv(RESULTS_DIR / "e2" / "follow.csv", rows)
    outside = has_c_star & ((gap < GAP_EDGES[0]) | (gap >= GAP_EDGES[-1]))
    write_csv(RESULTS_DIR / "e2" / "summary.csv", [{
        "episodes": len(gap), "without_c_star": int((~has_c_star).sum()),
        "outside_plotted_range": int(outside.sum()),
        "median_c_star": round(float(np.nanmedian(ep["c_star"])), 4),
    }])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for key, label in METHODS:
        if not curves.get(key):
            continue
        x, m, h, _ = map(np.array, zip(*curves[key]))
        line, = axes[0].plot(x, m, "-o" if key.startswith("ppo") else "--", markersize=3, label=label)
        axes[0].fill_between(x, m - h, m + h, color=line.get_color(), alpha=0.2)
    for name, pts in split.items():
        if not pts:
            continue
        x, m, h, _ = map(np.array, zip(*pts))
        line, = axes[1].plot(x, m, "-o", markersize=3, label=f"PPO-Conf, {name} claimed room")
        axes[1].fill_between(x, m - h, m + h, color=line.get_color(), alpha=0.2)
    for key, label in METHODS:
        pts = [r for r in rows if r["panel"] == "c" and r["method"] == label]
        if pts:
            axes[2].errorbar([r["bin_center"] for r in pts], [r["follow_rate"] for r in pts],
                             yerr=[r["ci95"] for r in pts], marker="o", markersize=3, capsize=2, label=label)
    for ax in axes[:2]:
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("c - c*")
    axes[2].set_xlabel("displayed confidence c")
    titles = ("(a) first search is the claimed room", "(b) PPO-Conf by distance of the claim",
              "(c) trust curve by confidence bin")
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("follow rate")
    save_figure(fig, "e2_trust")
    plt.close(fig)


if __name__ == "__main__":
    main()
