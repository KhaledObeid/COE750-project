"""ProcTHOR-10K row (DECISIONS "Two last analyses", 1; 3a optional row): convert and evaluate.

    pip install prior==1.0.3                     # only needed for this row
    python -m trustmap.procthor convert          # -> data/procthor/houses.json, results/procthor/skips.csv, houses.png
    python -m trustmap.procthor evaluate         # -> results/procthor/table.csv, results/figures/procthor_regret

ProcTHOR-10K (Deitke et al., NeurIPS 2022) is a public set of 10,000
procedurally generated houses. This module takes the houses with exactly 5
rooms, in dataset order, from the test split and then the validation split,
until 100 valid houses are converted, and redraws each one with our own rules
(the same ones as trustmap.generator):

  - occupancy grid of 0.10 m cells; floor = inside a room polygon;
  - walls 0.20 m thick, centred on ProcTHOR's own wall segments (two cells, the
    same half-open rule as our generator), except walls ProcTHOR marks "empty"
    (open connections, e.g. kitchen to living room);
  - door openings cut where ProcTHOR's doors are (its holePolygon along the
    wall); the door to the outside stays closed, since outside is blocked;
  - planning grid = walls inflated by the robot radius 0.20 m (grid.inflate);
  - viewpoint = the free planning cell of the room nearest the room centre
    (here the centroid of the room's floor cells, since rooms can be L-shaped);
  - 8-connected A*/Dijkstra distance table between viewpoints (planning.py);
  - object placement rule of Q4 (generator.object_candidates, camera.py).
Room types: Kitchen, LivingRoom, Bedroom, Bathroom -> kitchen, living, bedroom,
bathroom; the co-occurrence weights apply to whatever rooms are present.
A house is skipped if any rule fails (a room with no free cell or unreachable
after inflation, a room with no valid object cell); skips are counted by reason.

Evaluation (no training): the 15 main policies (PPO-Conf, PPO-Binary,
PPO-NoPrior, 5 seeds each) and the four baselines, plus the optimum as the
reference, at kappa 1, lambda 0.5, table placement, rho in 9a, 25 episodes per
house per rho with seed 300,000; same metrics and t-intervals as E1.
Converted houses get ids 1,000,000 + index (test split) and 1,100,000 + index
(validation split), so they can never be confused with our seed ranges.
"""

import argparse
import json
import math
import multiprocessing
import os
from collections import Counter
from dataclasses import dataclass

import numpy as np

from trustmap.experiments import (
    BASELINES,
    OPTIMUM,
    RESULTS_DIR,
    TEST_RHOS,
    Condition,
    Records,
    play_records,
    ppo_methods,
    save_figure,
    t_interval,
    write_csv,
)
from trustmap.generator import (
    CELL_SIZE_M,
    CONNECTIVITY,
    ROBOT_RADIUS_M,
    WALL_THICKNESS_M,
    object_candidates,
)
from trustmap.grid import OccupancyGrid, inflate
from trustmap.house import House
from trustmap.planning import distance_table

REPO = RESULTS_DIR.parent
DATA = REPO / "data" / "procthor" / "houses.json"
OUT = RESULTS_DIR / "procthor"
TARGET_HOUSES = 100
REVISION = "439193522244720b86d8c81cde2e51e3a4d150cf"  # procthor-10k git commit (main on 2026-11-19)
EPISODE_SEED = 300_000      # DECISIONS "Two last analyses"
EPISODES_PER_HOUSE = 25
ID_OFFSET = {"test": 1_000_000, "val": 1_100_000}
TYPE_MAP = {"Kitchen": "kitchen", "LivingRoom": "living", "Bedroom": "bedroom", "Bathroom": "bathroom"}
MARGIN_M = 0.4              # free border of blocked cells around the house on the grid
HALF_WALL = WALL_THICKNESS_M / 2


class Skip(Exception):
    """This house breaks a rule; the message is the reason."""


@dataclass(frozen=True)
class ProcthorLayout:
    """Same interface as generator.HouseLayout where the shared code needs it."""

    seed: int
    room_types: tuple
    polygons: tuple       # per room: list of (x, y) corners in meters
    grid: OccupancyGrid   # walls only (camera)
    planning_grid: OccupancyGrid
    cells: tuple          # per room: list of (row, col) floor cells (not wall cells)
    viewpoints: tuple
    house: House = None

    @property
    def rooms(self):
        return self.polygons

    def viewpoint_xy(self, room):
        return self.grid.cell_to_world(self.viewpoints[room])

    def room_cells(self, room):
        return self.cells[room]


# ------------------------------------------------------------ conversion ----

def _segment(points):
    """The two distinct floor-level (x, z) end points of a ProcTHOR wall polygon, in order."""
    ends = []
    for p in points:
        xy = (round(p["x"], 6), round(p["z"], 6))
        if xy not in ends:
            ends.append(xy)
    if len(ends) != 2:
        raise Skip("wall polygon is not a vertical rectangle")
    return ends


def _band(centres, a, b):
    """Index mask of cell centres in the half-open interval [a, b)."""
    return (centres >= a) & (centres < b)


def convert(h, seed):
    """One ProcTHOR house (its JSON dict) -> ProcthorLayout, or raise Skip(reason)."""
    from matplotlib.path import Path as MplPath

    rooms = h["rooms"]
    if len(rooms) != 5:
        raise Skip("not 5 rooms")
    types = tuple(TYPE_MAP.get(r["roomType"], "?") for r in rooms)
    if "?" in types:
        raise Skip("room type outside Kitchen, LivingRoom, Bedroom, Bathroom")
    polys = [[(p["x"], p["z"]) for p in r["floorPolygon"]] for r in rooms]
    allp = np.array([p for poly in polys for p in poly])
    x_min, y_min = allp.min(0) - MARGIN_M
    x_max, y_max = allp.max(0) + MARGIN_M
    cs = CELL_SIZE_M
    cols, rows = int(math.ceil((x_max - x_min) / cs)), int(math.ceil((y_max - y_min) / cs))
    xc = x_min + (np.arange(cols) + 0.5) * cs   # cell centre x per column
    yc = y_min + (np.arange(rows) + 0.5) * cs   # cell centre y per row
    gx, gy = np.meshgrid(xc, yc)
    pts = np.column_stack([gx.ravel(), gy.ravel()])

    inside = [MplPath(np.array(poly)).contains_points(pts).reshape(rows, cols) for poly in polys]
    occupied = ~np.any(inside, axis=0)

    # Walls: 0.20 m bands centred on every non-empty wall segment, extended by half a wall at the ends
    # so corners close. ProcTHOR stores each wall once per room side; both sides give the same band.
    wall = np.zeros_like(occupied)
    wall_v, wall_h = np.zeros_like(occupied), np.zeros_like(occupied)  # bands of vertical / horizontal walls
    walls_by_id = {}
    for w in h["walls"]:
        (x0, y0), (x1, y1) = _segment(w["polygon"])
        walls_by_id[w["id"]] = ((x0, y0), (x1, y1))
        if w.get("empty"):
            continue
        if abs(x0 - x1) < 1e-6:      # vertical wall, constant x
            c = _band(xc, x0 - HALF_WALL, x0 + HALF_WALL)
            r = _band(yc, min(y0, y1) - HALF_WALL, max(y0, y1) + HALF_WALL)
        elif abs(y0 - y1) < 1e-6:    # horizontal wall, constant y
            r = _band(yc, y0 - HALF_WALL, y0 + HALF_WALL)
            c = _band(xc, min(x0, x1) - HALF_WALL, max(x0, x1) + HALF_WALL)
        else:
            raise Skip("wall not axis-aligned")
        band = np.outer(r, c)
        wall |= band
        if abs(x0 - x1) < 1e-6:
            wall_v |= band
        else:
            wall_h |= band
    occupied |= wall

    # Doors: cut the opening through the wall band. holePolygon x is measured along wall0 from its first point.
    for d in h["doors"]:
        if "exterior" in d["wall0"] or "exterior" in d["wall1"]:
            continue  # the entrance door: outside is blocked anyway
        (x0, y0), (x1, y1) = walls_by_id[d["wall0"]]
        # Offsets are measured from the wall's first polygon point (ProcTHOR's p0).
        a, b = sorted(p["x"] for p in d["holePolygon"][:2])
        length = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        (sx, sy), (ex, ey) = (x0 + ux * a, y0 + uy * a), (x0 + ux * b, y0 + uy * b)
        if abs(x0 - x1) < 1e-6:
            c = _band(xc, x0 - HALF_WALL, x0 + HALF_WALL)
            r = _band(yc, min(sy, ey), max(sy, ey))
        else:
            r = _band(yc, y0 - HALF_WALL, y0 + HALF_WALL)
            c = _band(xc, min(sx, ex), max(sx, ex))
        # Clear only this wall's band: where a door runs up to a corner, the band of the crossing wall
        # stays closed (checked by trustmap.procthor_check).
        crossing = wall_h if abs(x0 - x1) < 1e-6 else wall_v
        occupied[np.outer(r, c) & ~crossing] = False

    grid = OccupancyGrid(occupied, cs, (float(x_min), float(y_min)))
    planning = inflate(grid, ROBOT_RADIUS_M)

    cells, viewpoints = [], []
    for k in range(5):
        floor = inside[k] & ~wall
        rc = [tuple(int(v) for v in x) for x in np.argwhere(floor)]
        if not rc:
            raise Skip("room has no floor cell")
        cy, cx = np.mean(rc, axis=0) + 0.5  # room centre in grid units (centroid of its floor cells)
        free = [c for c in rc if planning.is_free(c)]
        if not free:
            raise Skip("room has no free cell after inflation")
        viewpoints.append(min(free, key=lambda c: ((c[1] + 0.5 - cx) ** 2 + (c[0] + 0.5 - cy) ** 2, c)))
        cells.append(rc)
    try:
        distances = distance_table(planning, viewpoints, CONNECTIVITY)
    except ValueError:
        raise Skip("room unreachable after inflation")
    house = House(room_types=types, distances=distances, seed=seed)
    layout = ProcthorLayout(seed, types, tuple(polys), grid, planning, tuple(cells), tuple(viewpoints), house)
    for k in range(5):
        if not object_candidates(layout, k):
            raise Skip("no valid object cell")
    return layout


def _convert_job(job):
    split, index, h = job
    try:
        return split, index, convert(h, ID_OFFSET[split] + index), None
    except Skip as e:
        return split, index, None, str(e)


def load_procthor():
    import prior

    # Pinned to the revision that was "main" when the row was computed (Q14), so reruns use the same houses.
    return prior.load_dataset("procthor-10k", revision=REVISION)


def convert_all(workers=None):
    """Convert 5-room houses in dataset order (test, then validation) until TARGET_HOUSES are valid."""
    data = load_procthor()
    jobs = [(split, i, h) for split in ("test", "val") for i, h in enumerate(data[split]) if len(h["rooms"]) == 5]
    with multiprocessing.Pool(workers or os.cpu_count()) as pool:
        results = pool.map(_convert_job, jobs)  # keeps dataset order
    valid, skips = [], []
    for split, i, layout, reason in results:
        if len(valid) >= TARGET_HOUSES:
            break
        if layout is None:
            skips.append({"split": split, "index": i, "reason": reason})
        else:
            valid.append((split, i, layout))
    return valid, skips


def save_pool(valid):
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps([{"id": lay.seed, "split": split, "index": i, "room_types": list(lay.room_types),
                                 "distances": np.round(lay.house.distances, 6).tolist()}
                                for split, i, lay in valid], indent=1))
    print(f"  wrote {DATA.relative_to(REPO)} ({len(valid)} houses)")


def load_houses():
    return [House(room_types=tuple(h["room_types"]), distances=np.array(h["distances"]), seed=h["id"])
            for h in json.loads(DATA.read_text())]


def plot_houses(valid, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(valid), figsize=(4.2 * len(valid), 4.6))
    for ax, (split, i, lay) in zip(np.atleast_1d(axes), valid):
        g = lay.grid
        img = np.where(g.occupied, 0.15, np.where(lay.planning_grid.occupied, 0.65, 1.0))
        ext = (g.origin[0], g.origin[0] + g.shape[1] * g.cell_size, g.origin[1], g.origin[1] + g.shape[0] * g.cell_size)
        ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
        for k in range(5):
            cand = object_candidates(lay, k)
            xy = np.array([g.cell_to_world(c) for c in cand])
            ax.scatter(xy[:, 0], xy[:, 1], s=0.6, marker="s", linewidths=0, color="tab:orange", alpha=0.35)
            vx, vy = lay.viewpoint_xy(k)
            ax.plot(vx, vy, "o", color="tab:blue", markersize=5)
            ax.annotate(lay.room_types[k], (vx, vy), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8, color="tab:blue")
        ax.set_title(f"ProcTHOR-10K {split} #{i} (id {lay.seed})", fontsize=9)
        ax.set_xlabel("x (m)")
        ax.set_aspect("equal")
    fig.suptitle("Converted ProcTHOR houses: walls (black), inflation (grey), viewpoints (blue), "
                 "valid object cells (orange)", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ------------------------------------------------------------ evaluation ----

CONDITION = Condition("procthor")  # kappa 1, lambda 0.5, table placement
LABELS = {"random": "Random", "nearest": "Nearest-unexplored", "follower": "Greedy map follower",
          "bayes_nominal": "Bayes-nominal", "optimum": "Optimum", "ppo_conf": "PPO-Conf",
          "ppo_binary": "PPO-Binary", "ppo_noprior": "PPO-NoPrior"}
ORDER = ("random", "nearest", "follower", "bayes_nominal", "ppo_noprior", "ppo_binary", "ppo_conf", "optimum")
METRICS = ("regret_m", "success", "spl", "success_first_decision", "distance_m")


def methods():
    out = [*BASELINES, OPTIMUM]
    for key, label, vk in (("ppo_conf", "PPO-Conf", "conf"), ("ppo_binary", "PPO-Binary", "binary"),
                           ("ppo_noprior", "PPO-NoPrior", "noprior")):
        out += ppo_methods(key, label, vk, range(5))
    return out


def records_path(rho):
    return OUT / f"records_rho{rho:g}.npz"


def _evaluate_rho(rho):
    from trustmap.evaluation import make_test_episodes, solve_all

    cfg = CONDITION.config(rho)
    episodes = make_test_episodes(load_houses(), cfg, EPISODES_PER_HOUSE, EPISODE_SEED)
    solutions = solve_all(episodes, cfg.max_decisions)
    data = {"episode/house_seed": np.array([h.seed for h, _ in episodes])}
    for m in methods():
        for f, v in play_records(m, episodes, solutions, EPISODE_SEED).items():
            data[f"{m.key}/{m.seed}/{f}"] = v
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(records_path(rho), **data)
    return records_path(rho)


def load(rho):
    data = np.load(records_path(rho))
    rec = Records({k.split("/", 1)[1]: data[k] for k in data.files if k.startswith("episode/")})
    for k in data.files:
        if not k.startswith("episode/"):
            key, seed, fld = k.split("/")
            rec.runs.setdefault((key, None if seed == "None" else int(seed)), {})[fld] = data[k]
    return rec


def table_rows():
    rows = []
    for rho in TEST_RHOS:
        rec = load(rho)
        keys = sorted({k for k, _ in rec.runs}, key=lambda k: ORDER.index(k))
        for key in keys:
            seeds = rec.seeds(key)
            row = {"condition": "procthor", "rho": rho, "method": LABELS[key],
                   "seeds": 0 if seeds == [None] else len(seeds),
                   "houses": len(set(rec.episode["house_seed"])), "episodes": len(rec.episode["house_seed"])}
            for m in METRICS:
                mean, half = t_interval(rec.per_seed(key, m))
                row[m] = round(mean, 4)
                row[f"{m}_ci95"] = round(half, 4)
            rows.append(row)
    return rows


def figure(rows, n_houses):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, title in ((axes[0], "regret_m", "Decision regret (m, lower is better)"),
                              (axes[1], "success", "Success rate")):
        for label in dict.fromkeys(r["method"] for r in rows):
            pts = [r for r in rows if r["method"] == label]
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
    fig.suptitle(f"ProcTHOR-10K row: {n_houses} converted houses, no training (policies trained on our houses)",
                 fontsize=10)
    save_figure(fig, "procthor_regret")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["convert", "evaluate", "figures"])
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args()
    if a.step == "convert":
        valid, skips = convert_all(a.workers)
        save_pool(valid)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "dataset_revision.txt").write_text(f"procthor-10k {REVISION}\n")
        counts = Counter(s["reason"] for s in skips)
        write_csv(OUT / "skips.csv", skips or [{"split": "", "index": "", "reason": "none"}])
        write_csv(OUT / "skip_counts.csv", [{"reason": k, "houses": v} for k, v in counts.items()]
                  or [{"reason": "none", "houses": 0}])
        print(f"valid {len(valid)} (test {sum(s == 'test' for s, _, _ in valid)}, "
              f"val {sum(s == 'val' for s, _, _ in valid)}); skipped {len(skips)}: {dict(counts)}")
        plot_houses(valid[:4], OUT / "houses.png")
        return
    if a.step == "evaluate":
        print(f"procthor: {len(methods())} method runs x {len(TEST_RHOS)} rho values")
        with multiprocessing.Pool(a.workers or len(TEST_RHOS)) as pool:
            for path in pool.imap_unordered(_evaluate_rho, TEST_RHOS):
                print(f"  saved {path.relative_to(REPO)}")
    rows = table_rows()
    write_csv(OUT / "table.csv", rows)
    figure(rows, rows[0]["houses"])


if __name__ == "__main__":
    main()
