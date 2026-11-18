"""E1: main comparison across rho (Point 9a, 9b), with H1 and H4 (Point 10a).

    python -m trustmap.e1                    # evaluate, then write tables and figures
    python -m trustmap.e1 --figures-only     # redraw from the saved records

Conditions:
  main     kappa 1, lambda 0.5, table placement. Methods: the four baselines, the
           optimum (reference), PPO-Conf, PPO-Binary and PPO-NoPrior (5 seeds each).
           The records also hold c* of every episode, used by E2.
  uniform  the uniform-placement ablation row (3b): world with no commonsense,
           baselines plus PPO-Conf trained with placement = uniform (3 seeds).
  lam0, lam1  lambda sensitivity row (3c). PLACEHOLDER (OPEN_QUESTIONS Q10):
           which policies this row trains is not decided; for now the main
           policies are tested at lambda 0 and 1. Run only on request.

Outputs (results/e1/): one records file per condition and rho, table.csv
(mean and 95% half-width per method, rho and metric), h4.csv; figures
results/figures/e1_regret and h4_habit_gap.
"""

import argparse

import numpy as np

from trustmap.experiments import (
    BASELINES,
    OPTIMUM,
    TEST_RHOS,
    Condition,
    evaluate_condition,
    load_records,
    ppo_methods,
    records_path,
    save_figure,
    t_interval,
    write_csv,
)

CONDITIONS = {
    "main": Condition("main"),
    "uniform": Condition("uniform", placement="uniform"),
    "lam0": Condition("lam0", lam=0.0),
    "lam1": Condition("lam1", lam=1.0),
}
METRICS = ("regret_m", "success", "spl", "success_first_decision", "distance_m")
PPO_MAIN = (("ppo_conf", "PPO-Conf", "conf"), ("ppo_binary", "PPO-Binary", "binary"),
            ("ppo_noprior", "PPO-NoPrior", "noprior"))


def methods_for(name):
    if name == "uniform":
        return [*BASELINES, OPTIMUM, *ppo_methods("ppo_conf", "PPO-Conf (uniform)", "conf", range(3),
                                                   placement="uniform")]
    methods = [*BASELINES, OPTIMUM]
    for key, label, vk in PPO_MAIN:
        methods += ppo_methods(key, label, vk, range(5))
    return methods


LABELS = {"random": "Random", "nearest": "Nearest-unexplored", "follower": "Greedy map follower",
          "bayes_nominal": "Bayes-nominal", "optimum": "Optimum", "ppo_conf": "PPO-Conf",
          "ppo_binary": "PPO-Binary", "ppo_noprior": "PPO-NoPrior"}
ORDER = ("random", "nearest", "follower", "bayes_nominal", "ppo_noprior", "ppo_binary", "ppo_conf", "optimum")


def table_rows(condition):
    rows = []
    for rho in TEST_RHOS:
        rec = load_records("e1", condition, rho)
        keys = sorted({k for k, _ in rec.runs}, key=lambda k: ORDER.index(k) if k in ORDER else 99)
        for key in keys:
            row = {"condition": condition.name, "rho": rho, "method": LABELS.get(key, key),
                   "seeds": len(rec.seeds(key)) if rec.seeds(key) != [None] else 0}
            for m in METRICS:
                mean, half = t_interval(rec.per_seed(key, m))
                row[m] = round(mean, 4)
                row[f"{m}_ci95"] = round(half, 4)
            rows.append(row)
    return rows


def h4_rows():
    """H4: PPO-NoPrior regret minus PPO-Conf regret, for weak- and strong-habit targets.

    Seeds are paired by index (NoPrior seed k with Conf seed k); the interval is
    a t-interval over these paired differences.
    """
    rows = []
    main = CONDITIONS["main"]
    for rho in TEST_RHOS:
        rec = load_records("e1", main, rho)
        for group, mask in (("weak habit", rec.episode["weak_habit"]), ("strong habit", ~rec.episode["weak_habit"])):
            noprior = rec.per_seed("ppo_noprior", "regret_m", mask)
            conf = rec.per_seed("ppo_conf", "regret_m", mask)
            n = min(len(noprior), len(conf))
            mean, half = t_interval(noprior[:n] - conf[:n])
            rows.append({"rho": rho, "group": group, "pairs": n, "gap_m": round(mean, 4),
                         "gap_ci95": round(half, 4), "episodes": int(mask.sum())})
    return rows


def figures(rows, h4):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = [r for r in rows if r["condition"] == "main"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, title in ((axes[0], "regret_m", "Decision regret (m, lower is better)"),
                              (axes[1], "success", "Success rate")):
        for label in dict.fromkeys(r["method"] for r in main):
            pts = [r for r in main if r["method"] == label]
            x = np.array([r["rho"] for r in pts])
            y = np.array([r[metric] for r in pts])
            h = np.array([r[f"{metric}_ci95"] for r in pts])
            style = "-o" if label.startswith("PPO") else "--s" if label != "Optimum" else ":"
            line, = ax.plot(x, y, style, label=label, markersize=4)
            if h.any():
                ax.fill_between(x, y - h, y + h, color=line.get_color(), alpha=0.2)
        ax.set_xlabel("corruption level rho")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    save_figure(fig, "e1_regret")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    width = 0.07
    for i, group in enumerate(("strong habit", "weak habit")):
        pts = [r for r in h4 if r["group"] == group]
        x = np.array([r["rho"] for r in pts]) + (i - 0.5) * width
        ax.bar(x, [r["gap_m"] for r in pts], width, yerr=[r["gap_ci95"] for r in pts], capsize=3, label=group)
    ax.set_xlabel("corruption level rho")
    ax.set_ylabel("regret(PPO-NoPrior) - regret(PPO-Conf) (m)")
    ax.set_title("H4: value of the map by habit strength", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    save_figure(fig, "h4_habit_gap")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", nargs="+", choices=list(CONDITIONS), default=["main", "uniform"])
    p.add_argument("--figures-only", action="store_true", help="skip evaluation, redraw from saved records")
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args()

    if not a.figures_only:
        for name in a.condition:
            cond = CONDITIONS[name]
            evaluate_condition("e1", cond, TEST_RHOS, methods_for(name), with_c_star=(name == "main"),
                               workers=a.workers)

    from trustmap.experiments import RESULTS_DIR

    names = [n for n in CONDITIONS if all(records_path("e1", CONDITIONS[n], r).exists() for r in TEST_RHOS)]
    rows = [row for n in names for row in table_rows(CONDITIONS[n])]
    write_csv(RESULTS_DIR / "e1" / "table.csv", rows)
    h4 = h4_rows()
    write_csv(RESULTS_DIR / "e1" / "h4.csv", h4)
    figures(rows, h4)


if __name__ == "__main__":
    main()
