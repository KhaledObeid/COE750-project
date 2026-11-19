"""Conversion check of the ProcTHOR-10K row (DECISIONS "Answers to ... Q13-Q14 and a ProcTHOR conversion check").

    python -m trustmap.procthor_check    # -> results/procthor/conversion_check.csv, conversion_check_openings.csv

For each of the 100 converted houses (data/procthor/houses.json), the house is
converted again from ProcTHOR's own JSON and checked against that JSON:

  1. Openings. Along every boundary that two rooms share (from their floor
     polygons, whether or not ProcTHOR lists a wall there), a position is
     "open" when every cell of the 0.20 m wall band across the boundary is free.
     Each run of open positions must match exactly one opening that ProcTHOR
     lists on that boundary: a door's hole, or a wall marked "empty" (open
     connection). Its width must equal the listed width within one grid cell
     (0.10 m). The listed width is first reduced by the parts covered by the
     band of a crossing wall (a wall meeting the boundary at a corner), since
     those cells are wall in ProcTHOR too.
  1b. No free cell lies outside every room polygon (nothing opens to the outside).
  2. Walls. Every position of every wall ProcTHOR lists (not "empty") is closed,
     except inside a listed interior door (plus one cell of tolerance at each
     door edge). The entrance door to the outside must stay closed.
  3. Objects. Every valid object cell lies inside its own room's polygon.
"""

import csv
import json
import multiprocessing
import os

import numpy as np
from matplotlib.path import Path as MplPath

from trustmap.generator import CELL_SIZE_M, object_candidates
from trustmap.procthor import DATA, HALF_WALL, OUT, convert, load_procthor

TOL = CELL_SIZE_M + 1e-6   # "within one grid cell"
EPS = 1e-6


def _key(v):
    return round(float(v), 4)


def _segments(h):
    """Every wall as (id, vertical, line, lo, hi, empty)."""
    out = []
    for w in h["walls"]:
        pts = []
        for p in w["polygon"]:
            xy = (_key(p["x"]), _key(p["z"]))
            if xy not in pts:
                pts.append(xy)
        (x0, y0), (x1, y1) = pts
        vertical = abs(x0 - x1) < EPS
        line, lo, hi = (x0, min(y0, y1), max(y0, y1)) if vertical else (y0, min(x0, x1), max(x0, x1))
        out.append({"id": w["id"], "vertical": vertical, "line": line, "lo": lo, "hi": hi,
                    "empty": bool(w.get("empty")), "p0": pts[0], "p1": pts[1]})
    return out


def _edges(poly):
    """Axis-aligned polygon edges as (vertical, line, lo, hi)."""
    out = []
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        x0, y0, x1, y1 = map(_key, (x0, y0, x1, y1))
        if abs(x0 - x1) < EPS:
            out.append((True, x0, min(y0, y1), max(y0, y1)))
        else:
            out.append((False, y0, min(x0, x1), max(x0, x1)))
    return out


def _shared_boundaries(polys):
    """(room a, room b, vertical, line, lo, hi) for every boundary piece two rooms share."""
    edges = [_edges(p) for p in polys]
    out = []
    for a in range(len(polys)):
        for b in range(a + 1, len(polys)):
            for va, la, loa, hia in edges[a]:
                for vb, lb, lob, hib in edges[b]:
                    if va == vb and abs(la - lb) < EPS:
                        lo, hi = max(loa, lob), min(hia, hib)
                        if hi - lo > EPS:
                            out.append((a, b, va, la, lo, hi))
    return out


def _door_intervals(h, segs):
    """Interior doors as (vertical, line, lo, hi, door id, asset); exterior doors separately."""
    by_id = {s["id"]: s for s in segs}
    inner, outer = [], []
    for d in h["doors"]:
        s = by_id[d["wall0"]]
        (x0, y0), (x1, y1) = s["p0"], s["p1"]
        length = np.hypot(x1 - x0, y1 - y0)
        a, b = sorted(p["x"] for p in d["holePolygon"][:2])
        # Offsets from the wall's first polygon point (Q14).
        ends = [(x0 + (x1 - x0) * t / length, y0 + (y1 - y0) * t / length) for t in (a, b)]
        coord = [e[1] if s["vertical"] else e[0] for e in ends]
        item = (s["vertical"], s["line"], min(coord), max(coord), d["id"], d["assetId"])
        (outer if "exterior" in d["wall0"] or "exterior" in d["wall1"] else inner).append(item)
    return inner, outer


def _subtract(intervals, cuts):
    """Intervals minus the union of cut intervals."""
    out = list(intervals)
    for c0, c1 in cuts:
        nxt = []
        for a, b in out:
            if c1 <= a or c0 >= b:
                nxt.append((a, b))
                continue
            if c0 > a:
                nxt.append((a, c0))
            if c1 < b:
                nxt.append((c1, b))
        out = nxt
    return [(a, b) for a, b in out if b - a > EPS]


def _band_cells(grid, vertical, line, pos):
    """The grid cells of the 0.20 m wall band across the line at position pos (a cell centre along the line)."""
    xc0, yc0 = grid.origin
    cs = grid.cell_size
    if vertical:
        cols = [c for c in range(grid.shape[1]) if line - HALF_WALL <= xc0 + (c + 0.5) * cs < line + HALF_WALL]
        row = int(np.floor((pos - yc0) / cs))
        return [(row, c) for c in cols]
    rows = [r for r in range(grid.shape[0]) if line - HALF_WALL <= yc0 + (r + 0.5) * cs < line + HALF_WALL]
    col = int(np.floor((pos - xc0) / cs))
    return [(r, col) for r in rows]


def _positions(grid, vertical, lo, hi):
    """Cell-centre coordinates along the line strictly inside (lo, hi)."""
    o = grid.origin[1] if vertical else grid.origin[0]
    cs = grid.cell_size
    k0, k1 = int(np.ceil((lo - o) / cs - 0.5)), int(np.floor((hi - o) / cs - 0.5))
    return [o + (k + 0.5) * cs for k in range(k0, k1 + 1) if lo < o + (k + 0.5) * cs < hi]


def _runs(flags, pos):
    """Contiguous runs of True as (start, end) in meters (cell edges)."""
    runs, start = [], None
    cs = CELL_SIZE_M
    for f, p in zip(flags + [False], pos + [None]):
        if f and start is None:
            start = p
            last = p
        elif f:
            last = p
        elif start is not None:
            runs.append((start - cs / 2, last + cs / 2))
            start = None
    return runs


def check_house(job):
    split, index, hid, h = job
    lay = convert(h, hid)
    grid = lay.grid
    polys = [list(p) for p in lay.polygons]
    segs = _segments(h)
    doors, ext_doors = _door_intervals(h, segs)
    solid = [s for s in segs if not s["empty"]]
    openings, unmatched, width_errors = [], 0, []

    # 1. Openings along every shared boundary.
    for a, b, vertical, line, lo, hi in _shared_boundaries(polys):
        listed = [(max(lo, d[2]), min(hi, d[3]), f"door {d[4]} ({d[5]})") for d in doors
                  if d[0] == vertical and abs(d[1] - line) < EPS and min(hi, d[3]) - max(lo, d[2]) > EPS]
        empty = _subtract([(max(lo, s["lo"]), min(hi, s["hi"])) for s in segs if s["empty"] and s["vertical"] == vertical
                           and abs(s["line"] - line) < EPS and min(hi, s["hi"]) - max(lo, s["lo"]) > EPS], [])
        # merge overlapping empty pieces (both sides of an open wall are listed)
        empty_merged = []
        for e0, e1 in sorted(empty):
            if empty_merged and e0 <= empty_merged[-1][1] + EPS:
                empty_merged[-1] = (empty_merged[-1][0], max(empty_merged[-1][1], e1))
            else:
                empty_merged.append((e0, e1))
        listed += [(e0, e1, "open connection (empty wall)") for e0, e1 in empty_merged]
        # Bands of crossing walls (perpendicular, solid) along this line.
        cuts = [(s["line"] - HALF_WALL, s["line"] + HALF_WALL) for s in solid if s["vertical"] != vertical
                and s["lo"] - HALF_WALL <= line <= s["hi"] + HALF_WALL]
        pos = _positions(grid, vertical, lo, hi)
        flags = [all(grid.is_free(c) for c in _band_cells(grid, vertical, line, p)) for p in pos]
        runs = _runs(flags, pos)
        used = set()
        for r0, r1 in runs:
            match = [i for i, (e0, e1, _) in enumerate(listed) if min(r1, e1) - max(r0, e0) > EPS]
            row = {"split": split, "index": index, "id": hid, "rooms": f"{lay.room_types[a]}-{lay.room_types[b]}",
                   "line": f"{'x' if vertical else 'y'}={line}", "measured_from": round(r0, 3), "measured_to": round(r1, 3),
                   "measured_width_m": round(r1 - r0, 3)}
            if len(match) != 1:
                unmatched += 1
                row.update({"listed_as": "NOT LISTED" if not match else "several", "listed_width_m": "",
                            "clear_width_m": "", "width_error_m": "", "ok": False})
            else:
                e0, e1, what = listed[match[0]]
                used.add(match[0])
                clear = sum(y - x for x, y in _subtract([(e0, e1)], cuts))
                err = abs((r1 - r0) - clear)
                width_errors.append(err)
                row.update({"listed_as": what, "listed_width_m": round(e1 - e0, 3), "clear_width_m": round(clear, 3),
                            "width_error_m": round(err, 3), "ok": err <= TOL})
            openings.append(row)
        for i, (e0, e1, what) in enumerate(listed):
            clear = sum(y - x for x, y in _subtract([(e0, e1)], cuts))
            if i not in used and clear > TOL:  # a listed opening that is closed in the grid
                unmatched += 1
                openings.append({"split": split, "index": index, "id": hid,
                                 "rooms": f"{lay.room_types[a]}-{lay.room_types[b]}",
                                 "line": f"{'x' if vertical else 'y'}={line}", "measured_from": "", "measured_to": "",
                                 "measured_width_m": 0.0, "listed_as": what, "listed_width_m": round(e1 - e0, 3),
                                 "clear_width_m": round(clear, 3), "width_error_m": round(clear, 3), "ok": False})

    # 1b. Free cells outside every room polygon.
    rows, cols = grid.shape
    centres = np.array([grid.cell_to_world((r, c)) for r in range(rows) for c in range(cols)])
    inside_any = np.zeros(len(centres), bool)
    paths = [MplPath(np.array(p)) for p in polys]
    for pth in paths:  # a centre exactly on a polygon edge counts as inside (both radius signs)
        inside_any |= pth.contains_points(centres, radius=1e-6) | pth.contains_points(centres, radius=-1e-6)
    free_outside = int((~grid.occupied.ravel() & ~inside_any).sum())

    # 2. Listed walls closed except at interior doors.
    missing = 0
    for s in solid:
        holes = [(d[2] - CELL_SIZE_M, d[3] + CELL_SIZE_M) for d in doors
                 if d[0] == s["vertical"] and abs(d[1] - s["line"]) < EPS]
        for p in _positions(grid, s["vertical"], s["lo"], s["hi"]):
            if any(x <= p <= y for x, y in holes):
                continue
            if any(grid.is_free(c) for c in _band_cells(grid, s["vertical"], s["line"], p)):
                missing += 1
    ext_open = 0
    for v, line, lo, hi, _, _ in ext_doors:
        for p in _positions(grid, v, lo, hi):
            if all(grid.is_free(c) for c in _band_cells(grid, v, line, p)):
                ext_open += 1

    # 3. Object cells inside their own room polygon.
    n_obj, obj_out = 0, 0
    for k in range(5):
        cells = object_candidates(lay, k)
        n_obj += len(cells)
        xy = np.array([grid.cell_to_world(c) for c in cells])
        obj_out += int((~paths[k].contains_points(xy)).sum())

    n_ok = sum(1 for o in openings if o["ok"])
    summary = {
        "split": split, "index": index, "id": hid,
        "openings_measured": sum(1 for o in openings if o["measured_from"] != ""),
        "openings_listed_matched": n_ok, "openings_failed": len(openings) - n_ok,
        "doors_listed_interior": len(doors),
        "double_doorways": sum(1 for d in doors if "Double" in d[5]),
        "max_width_error_m": round(max(width_errors), 3) if width_errors else 0.0,
        "wall_positions_missing": missing, "exterior_door_positions_open": ext_open,
        "free_cells_outside_house": free_outside,
        "object_cells": n_obj, "object_cells_outside_own_room": obj_out,
    }
    summary["pass"] = (summary["openings_failed"] == 0 and missing == 0 and ext_open == 0
                       and free_outside == 0 and obj_out == 0)
    return summary, openings


def main():
    pool_houses = json.loads(DATA.read_text())
    data = load_procthor()
    jobs = [(h["split"], h["index"], h["id"], data[h["split"]][h["index"]]) for h in pool_houses]
    with multiprocessing.Pool(os.cpu_count()) as pool:
        results = pool.map(check_house, jobs)
    summaries = [s for s, _ in results]
    openings = [o for _, os_ in results for o in os_]
    for name, rows in (("conversion_check.csv", summaries), ("conversion_check_openings.csv", openings)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote results/procthor/{name} ({len(rows)} rows)")
    failed = [s for s in summaries if not s["pass"]]
    print(f"houses passing: {len(summaries) - len(failed)} / {len(summaries)}; openings checked: {len(openings)}, "
          f"failed: {sum(not o['ok'] for o in openings)}; max width error "
          f"{max(s['max_width_error_m'] for s in summaries)} m; missing wall positions "
          f"{sum(s['wall_positions_missing'] for s in summaries)}; free cells outside "
          f"{sum(s['free_cells_outside_house'] for s in summaries)}; object cells outside own room "
          f"{sum(s['object_cells_outside_own_room'] for s in summaries)}")
    for s in failed[:10]:
        print("  FAIL", s)


if __name__ == "__main__":
    main()
