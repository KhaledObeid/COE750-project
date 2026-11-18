"""E3: calibration study, the 2 x 2 of Point 9d, for H3.

    python -m trustmap.e3                    # evaluate, then write the table and figure
    python -m trustmap.e3 --figures-only

PPO-Conf trained at kappa 1 (5 seeds) and at kappa 3 (3 seeds) are each tested
at kappa 1 and kappa 3; Bayes-nominal (which takes c at face value) is tested
at both. rho in {0.2, 0.4, 0.6, 0.8}: at rho = 0 every map shows c = 1 whatever
kappa is, so it is left out. The matching cells (train kappa = test kappa)
show absorption of a consistently dishonest confidence; the crossed cells
show what happens when the dishonesty changes.

Fallback (Point 1, 9d): if the kappa-3 runs do not exist, the table and figure
contain only the kappa-1 policy tested at kappa 1 and 3 (evaluation-only E3).
Outputs: results/e3/table.csv, results/figures/e3_calibration.
"""

import argparse

import numpy as np

from trustmap.experiments import (
    OPTIMUM,
    RESULTS_DIR,
    Condition,
    Method,
    evaluate_condition,
    load_records,
    ppo_methods,
    records_path,
    save_figure,
    t_interval,
    write_csv,
)

RHOS = (0.2, 0.4, 0.6, 0.8)
TEST_CONDITIONS = (Condition("test_kappa1", kappa=1.0), Condition("test_kappa3", kappa=3.0))
LABELS = {"ppo_conf_train1": "PPO-Conf trained kappa 1", "ppo_conf_train3": "PPO-Conf trained kappa 3",
          "bayes_nominal": "Bayes-nominal", "optimum": "Optimum"}


def methods():
    return [
        *ppo_methods("ppo_conf_train1", LABELS["ppo_conf_train1"], "conf", range(5), kappa=1.0),
        *ppo_methods("ppo_conf_train3", LABELS["ppo_conf_train3"], "conf", range(3), kappa=3.0),
        Method("bayes_nominal", "Bayes-nominal"),
        OPTIMUM,
    ]


def table_rows():
    rows = []
    for cond in TEST_CONDITIONS:
        for rho in RHOS:
            if not records_path("e3", cond, rho).exists():
                continue
            rec = load_records("e3", cond, rho)
            for key in LABELS:
                if not rec.seeds(key):
                    continue
                mean, half = t_interval(rec.per_seed(key, "regret_m"))
                succ, succ_half = t_interval(rec.per_seed(key, "success"))
                rows.append({"test_kappa": cond.kappa, "rho": rho, "method": LABELS[key],
                             "seeds": 0 if rec.seeds(key) == [None] else len(rec.seeds(key)),
                             "regret_m": round(mean, 4), "regret_m_ci95": round(half, 4),
                             "success": round(succ, 4), "success_ci95": round(succ_half, 4)})
    return rows


def figure(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, kappa in zip(axes, (1.0, 3.0)):
        for label in ("PPO-Conf trained kappa 1", "PPO-Conf trained kappa 3", "Bayes-nominal"):
            pts = [r for r in rows if r["test_kappa"] == kappa and r["method"] == label]
            if not pts:
                continue
            x = np.array([r["rho"] for r in pts])
            y = np.array([r["regret_m"] for r in pts])
            h = np.array([r["regret_m_ci95"] for r in pts])
            line, = ax.plot(x, y, "-o" if label.startswith("PPO") else "--s", markersize=4, label=label)
            if h.any():
                ax.fill_between(x, y - h, y + h, color=line.get_color(), alpha=0.2)
        ax.set_title(f"tested at kappa = {kappa:g}", fontsize=10)
        ax.set_xlabel("corruption level rho")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("decision regret (m)")
    axes[0].legend(fontsize=8)
    save_figure(fig, "e3_calibration")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figures-only", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args()
    if not a.figures_only:
        for cond in TEST_CONDITIONS:
            evaluate_condition("e3", cond, RHOS, methods(), workers=a.workers)
    rows = table_rows()
    write_csv(RESULTS_DIR / "e3" / "table.csv", rows)
    figure(rows)


if __name__ == "__main__":
    main()
