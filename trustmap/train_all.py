"""Train every reported policy (Point 7d), several runs side by side on separate CPU cores.

    python -m trustmap.train_all --workers 7
    python -m trustmap.train_all --group main --workers 4

Groups:
  main     PPO-Conf, PPO-Binary, PPO-NoPrior at kappa 1, seeds 0-4 (E1, E2)      15 runs
  kappa3   PPO-Conf trained at kappa 3, seeds 0-2 (E3, 9d)                       3 runs
  uniform  PPO-Conf trained with placement = uniform, seeds 0-2 (E1 ablation)    3 runs

A run whose final_model.zip already exists is skipped, so the command can be
stopped and started again. Each run writes its log to runs/<name>/train.log.
The lambda sensitivity row trains nothing (Q10a: the main policies are tested
at lambda 0 and 1 in trustmap.e1), so the total is 21 runs.
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trustmap.train import run_name

GROUPS = {
    "main": [dict(variant=v, seed=s) for v in ("conf", "binary", "noprior") for s in range(5)],
    "kappa3": [dict(variant="conf", seed=s, kappa=3.0) for s in range(3)],
    "uniform": [dict(variant="conf", seed=s, placement="uniform") for s in range(3)],
}


def command(run, out):
    cmd = [sys.executable, "-m", "trustmap.train", "--variant", run["variant"], "--seed", str(run["seed"]),
           "--out", str(out)]
    if "kappa" in run:
        cmd += ["--kappa", str(run["kappa"])]
    if "placement" in run:
        cmd += ["--placement", run["placement"]]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    p.add_argument("--workers", type=int, default=4, help="runs trained at the same time (one CPU core each)")
    p.add_argument("--out", default="runs")
    a = p.parse_args()

    runs = [r for g in a.group for r in GROUPS[g]]
    todo = []
    for r in runs:
        name = run_name(r["variant"], r["seed"], r.get("kappa", 1.0), 0.5, r.get("placement", "table"))
        if (Path(a.out) / name / "final_model.zip").exists():
            print(f"skip {name} (already trained)")
        else:
            todo.append((name, r))

    def train_one(item):
        name, r = item
        log = Path(a.out) / name / "train.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            code = subprocess.call(command(r, a.out), stdout=f, stderr=subprocess.STDOUT)
        print(f"{'done' if code == 0 else 'FAILED'} {name}", flush=True)
        return code

    print(f"training {len(todo)} runs, {a.workers} at a time")
    with ThreadPoolExecutor(a.workers) as pool:
        codes = list(pool.map(train_one, todo))
    if any(codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
