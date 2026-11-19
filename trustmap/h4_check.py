"""H4 check: an analysis done AFTER seeing the H4 result (DECISIONS "Two last analyses", 2).

    python -m trustmap.h4_check     # -> results/e1/h4_check.csv, results/figures/h4_check.(pdf|png)

H4 predicted that the map helps PPO-Conf more (relative to PPO-NoPrior) for
weak-habit targets. E1 showed the opposite at rho >= 0.4. The explanation to
test: when the map claims the target's habit room with a low confidence, PPO-Conf
reads it as "probably not here", which PPO-NoPrior cannot see. If that is right,
the strong-habit advantage should come mostly from episodes whose claim is the
habit room.

Uses the existing E1 records (main condition, no new training or evaluation of
policies). The claimed room and the habit weights are not in the records, so the
same test episodes are regenerated from their seed (9b) and checked against the
records (house, target, c, r) before use.

Split: habit group (strong, weak) x whether the claimed room is the target's
most likely room by habit in that house (the room with the highest habit
weight; with a tie, any of the tied rooms counts). Gap = regret(PPO-NoPrior) -
regret(PPO-Conf), paired by seed index with a 95% t-interval over the 5 paired
differences, exactly as H4 (Q12.4).
"""

import numpy as np

from trustmap.e1 import CONDITIONS
from trustmap.evaluation import make_test_episodes
from trustmap.experiments import (
    EPISODES_PER_HOUSE,
    RESULTS_DIR,
    TEST_RHOS,
    TEST_SEED,
    load_records,
    save_figure,
    t_interval,
)
from trustmap.pool import load_pool

POST_HOC = "post hoc: analysis done after seeing the H4 result"


def claim_on_habit_room(rho, rec):
    """Boolean per E1 test episode: the claimed room has the highest habit weight in its house."""
    cfg = CONDITIONS["main"].config(rho)
    episodes = make_test_episodes(load_pool("test"), cfg, EPISODES_PER_HOUSE, TEST_SEED)
    # The regenerated episodes must be the recorded ones.
    assert np.array_equal([h.seed for h, _ in episodes], rec.episode["house_seed"])
    assert np.allclose([ep.c for _, ep in episodes], rec.episode["c"])
    assert np.allclose([ep.r for _, ep in episodes], rec.episode["r"])
    habit = np.array([ep.habit[ep.claimed_room] >= ep.habit.max() - 1e-12 for _, ep in episodes])
    return habit


def rows():
    out = []
    for rho in TEST_RHOS:
        rec = load_records("e1", CONDITIONS["main"], rho)
        on_habit = claim_on_habit_room(rho, rec)
        weak = rec.episode["weak_habit"]
        for group, gmask in (("strong habit", ~weak), ("weak habit", weak)):
            for claim, cmask in (("claim on habit room", on_habit), ("claim elsewhere", ~on_habit)):
                mask = gmask & cmask
                noprior = rec.per_seed("ppo_noprior", "regret_m", mask)
                conf = rec.per_seed("ppo_conf", "regret_m", mask)
                n = min(len(noprior), len(conf))
                gap, half = t_interval(noprior[:n] - conf[:n])
                c_mean, c_half = t_interval(conf[:n])
                np_mean, np_half = t_interval(noprior[:n])
                out.append({
                    "rho": rho, "group": group, "claim": claim, "episodes": int(mask.sum()),
                    "share_of_group": round(float(mask.sum() / gmask.sum()), 4),
                    "mean_c": round(float(rec.episode["c"][mask].mean()), 4),
                    "pairs": n, "gap_m": round(gap, 4), "gap_ci95": round(half, 4),
                    "conf_regret_m": round(c_mean, 4), "conf_ci95": round(c_half, 4),
                    "noprior_regret_m": round(np_mean, 4), "noprior_ci95": round(np_half, 4),
                })
    return out


def write(rows_, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# H4 check ({POST_HOC}). Gap = regret(PPO-NoPrior) - regret(PPO-Conf) in meters, "
                "paired by seed, 95% t-interval over 5 pairs. E1 main condition.\n")
        f.write(",".join(rows_[0]) + "\n")
        for r in rows_:
            f.write(",".join(str(v) for v in r.values()) + "\n")
    print(f"  wrote {path}")


def figure(rows_):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    width = 0.07
    for ax, group in zip(axes, ("strong habit", "weak habit")):
        for i, claim in enumerate(("claim on habit room", "claim elsewhere")):
            pts = [r for r in rows_ if r["group"] == group and r["claim"] == claim]
            x = np.array([r["rho"] for r in pts]) + (i - 0.5) * width
            ax.bar(x, [r["gap_m"] for r in pts], width, yerr=[r["gap_ci95"] for r in pts], capsize=3,
                   label=f"{claim} ({sum(r['episodes'] for r in pts) / len(pts):.0f} episodes per rho)")
        ax.set_title(f"{group} targets", fontsize=10)
        ax.set_xlabel("corruption level rho")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("regret(PPO-NoPrior) - regret(PPO-Conf) (m)")
    fig.suptitle(f"H4 check ({POST_HOC}): value of the map by habit strength and claimed room", fontsize=10)
    save_figure(fig, "h4_check")
    plt.close(fig)


def main():
    r = rows()
    write(r, RESULTS_DIR / "e1" / "h4_check.csv")
    figure(r)
    for x in r:
        print(x["rho"], x["group"], x["claim"], x["episodes"], x["gap_m"], x["gap_ci95"],
              x["conf_regret_m"], x["noprior_regret_m"])


if __name__ == "__main__":
    main()
