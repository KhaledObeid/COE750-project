"""Gazebo result rows (Point 9e, "Cost of a decision") from the runner's per-episode files.

    python -m trustmap.gazebo_summary                 # reads results/gazebo/house_*.csv

Writes results/gazebo/summary.csv with one line per policy and row:
  decision     success, SPL, decision regret (m), first-decision success and
               planned distance, all on planned (A*) distances, plus the share of
               episodes whose decisions equal the 2D simulator's (same_as_2d).
  end_to_end   driven distance, driving overhead (driven / planned - 1), Nav2
               failure rate (failed goals / goals; a goal fails after two
               attempts, Q11.4), end-to-end success (Q11.5) and the share of
               episodes where the virtual camera at the true pose agreed with
               the search rules.
Plain Python (csv only), so it also runs under ROS's Python.
"""

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def summarize(rows):
    out = []
    for policy in dict.fromkeys(r["policy"] for r in rows):
        eps = [r for r in rows if r["policy"] == policy]
        f = lambda k: [float(r[k]) for r in eps]  # noqa: E731
        goals = sum(f("decisions"))
        houses = " ".join(sorted({r["house"] for r in eps}))
        out.append({"policy": policy, "row": "decision", "houses": houses, "episodes": len(eps),
                    "success": round(mean(f("success")), 4), "spl": round(mean(f("spl")), 4),
                    "regret_m": round(mean(f("regret_m")), 4),
                    "success_first_decision": round(mean(f("success_first_decision")), 4),
                    "planned_m": round(mean(f("planned_m")), 3),
                    "same_as_2d": round(mean(f("same_as_2d")), 4)})
        planned, driven = sum(f("planned_m")), sum(f("driven_m"))
        out.append({"policy": policy, "row": "end_to_end", "houses": houses, "episodes": len(eps),
                    "driven_m": round(mean(f("driven_m")), 3),
                    "overhead_total": round(driven / planned - 1, 4) if planned else 0.0,
                    "overhead_mean": round(mean(f("overhead")), 4),
                    "nav_failure_rate": round(sum(f("nav_failures")) / goals, 4) if goals else 0.0,
                    "episodes_with_nav_failure": int(sum(1 for x in f("nav_failures") if x > 0)),
                    "end_to_end_success": round(mean(f("end_to_end_success")), 4),
                    "camera_agrees": round(mean(f("camera_agrees")), 4)})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=str(REPO / "results" / "gazebo"))
    a = p.parse_args()
    d = Path(a.dir)
    rows = []
    for path in sorted(d.glob("house_*.csv")):
        with open(path, newline="") as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"no house_*.csv in {d}")
    summary = summarize(rows)
    fields = list(dict.fromkeys(k for r in summary for k in r))
    with open(d / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    for r in summary:
        print(r)
    print(f"wrote {d / 'summary.csv'}")


if __name__ == "__main__":
    main()
