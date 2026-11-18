"""Plot houses to check them by eye (Point 3a, Q5).

    python -m trustmap.plot_house 0 1 2 3 --out houses.png

Shows walls (black), the area blocked by inflation for planning (grey), the
door openings, each room's type and viewpoint (dot), the cells where the
target may be placed (light dots, Q4), and the planned path from the first
room's viewpoint to every other room (A* on the inflated grid).
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # write files only, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from trustmap.generator import CONNECTIVITY, generate_house, object_candidates  # noqa: E402
from trustmap.planning import astar  # noqa: E402


def draw(ax, layout, show_objects=True):
    g = layout.grid
    rows, cols = g.shape
    extent = (g.origin[0], g.origin[0] + cols * g.cell_size, g.origin[1], g.origin[1] + rows * g.cell_size)
    # 0 = free floor, 1 = blocked only after inflation, 2 = wall.
    img = np.where(g.occupied, 2, np.where(layout.planning_grid.occupied, 1, 0))
    ax.imshow(img, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=2, interpolation="nearest")

    for k, room in enumerate(layout.rooms):
        x, y = layout.viewpoint_xy(k)
        if show_objects:
            pts = np.array([g.cell_to_world(c) for c in object_candidates(layout, k)])
            ax.plot(pts[:, 0], pts[:, 1], ".", color="tab:green", markersize=1, alpha=0.4)
        ax.plot(x, y, "o", color="tab:red", markersize=5)
        ax.annotate(f"{k} {room.type}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    for k in range(1, len(layout.rooms)):
        _, path = astar(layout.planning_grid, layout.viewpoints[0], layout.viewpoints[k], CONNECTIVITY)
        pts = np.array([g.cell_to_world(c) for c in path])
        ax.plot(pts[:, 0], pts[:, 1], "-", color="tab:blue", linewidth=0.8, alpha=0.7)

    ax.set_title(f"seed {layout.seed}: {layout.width_m:g} x {layout.depth_m:g} m, "
                 f"max distance {layout.house.max_distance:.1f} m", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seeds", type=int, nargs="+")
    p.add_argument("--out", default="houses.png")
    a = p.parse_args()
    n = len(a.seeds)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.6 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, seed in zip(axes.flat, a.seeds):
        draw(ax, generate_house(seed))
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
