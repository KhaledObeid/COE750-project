"""2D episode animation, the backup demo for the slides (Point 11b).

    python -m trustmap.animate_episode --house 200000 --episode 3 --policy follower --out demo.gif
    python -m trustmap.animate_episode --house 200000 --episode 3 --policy runs/<run>/best_model.zip --variant conf

Shows one test episode (the same E1 test episodes as the Gazebo runner, rho =
0.4): the map's claimed room and its confidence, the target, the rooms
already searched (grey), and the robot driving the planned A* path to each
room the policy chooses. A policy is "follower", "nearest", "random", an
exported .npz, or a MaskablePPO .zip with --variant.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from trustmap.baselines import GreedyMapFollower, NearestUnexplored, Random  # noqa: E402
from trustmap.episode import DEFAULT_MAX_DECISIONS  # noqa: E402
from trustmap.gazebo_runner import gazebo_episodes  # noqa: E402
from trustmap.generator import CONNECTIVITY, generate_house, place_object  # noqa: E402
from trustmap.planning import astar  # noqa: E402
from trustmap.search import RoomSearch  # noqa: E402

STEP_M = 0.25  # robot moves this far per animation frame


def load_policy(spec, variant):
    simple = {"follower": GreedyMapFollower, "nearest": NearestUnexplored, "random": Random}
    if spec in simple:
        return simple[spec]()
    if spec.endswith(".npz"):
        from trustmap.policy_export import NumpyPolicy

        return NumpyPolicy(spec)
    from sb3_contrib import MaskablePPO

    from trustmap.observation import VARIANT_KEYS
    from trustmap.ppo import PPOPolicy

    return PPOPolicy(MaskablePPO.load(spec, device="cpu"), VARIANT_KEYS[variant])


def frames_for(layout, ep, policy, rng):
    """Play the episode and return one frame description per animation step."""
    search = RoomSearch(layout.house, ep, DEFAULT_MAX_DECISIONS)
    frames = []
    xy = layout.viewpoint_xy(ep.start_room)
    note = f"start in the {layout.rooms[ep.start_room].type} (already searched)"
    frames += [(xy, search.searched.copy(), note)] * 8
    while not search.done:
        row = policy.choose(search, rng)
        room = int(ep.row_order[row])
        _, path = astar(layout.planning_grid, layout.viewpoints[search.current_room], layout.viewpoints[room],
                        CONNECTIVITY)
        pts = np.array([layout.grid.cell_to_world(c) for c in path])
        seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
        note = f"decision {search.decisions + 1}: search the {layout.rooms[room].type}"
        for d in np.arange(0, seg[-1], STEP_M):
            frames.append(((np.interp(d, seg, pts[:, 0]), np.interp(d, seg, pts[:, 1])), search.searched.copy(), note))
        search.step(row)
        found = search.success and room == ep.object_room
        result = "found it!" if found else ("not here: episode failed (3 searches)" if search.done else "not here")
        frames += [(tuple(pts[-1]), search.searched.copy(), f"{note}: {result}")] * 10
    return frames, search


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--house", type=int, default=200_000)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--policy", default="follower")
    p.add_argument("--variant", default="conf")
    p.add_argument("--out", default="episode.gif")
    a = p.parse_args()

    layout = generate_house(a.house)
    ep = gazebo_episodes(a.house, a.episode + 1)[a.episode]
    rng = np.random.default_rng([a.house, a.episode])
    target = place_object(layout, ep.object_room, rng)
    policy = load_policy(a.policy, a.variant)
    frames, search = frames_for(layout, ep, policy, rng)

    g = layout.grid
    rows, cols = g.shape
    extent = (g.origin[0], g.origin[0] + cols * g.cell_size, g.origin[1], g.origin[1] + rows * g.cell_size)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.where(g.occupied, 1, 0), origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1)
    shades = []
    for k, room in enumerate(layout.rooms):
        x0, y0, x1, y1 = (v * g.cell_size for v in room.rect)
        shade = Rectangle((x0, y0), x1 - x0, y1 - y0, color="grey", alpha=0.0)
        ax.add_patch(shade)
        shades.append(shade)
        label = room.type + (f"\nmap: here, c = {ep.c:.2f}" if k == ep.claimed_room else "")
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.6, label, ha="center", fontsize=8,
                color="tab:blue" if k == ep.claimed_room else "black")
    ax.plot(*target, "*", color="tab:red", markersize=14)
    robot, = ax.plot([], [], "o", color="tab:green", markersize=12)
    title = ax.set_title("", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"target: {ep.target.replace('_', ' ')}   policy: {getattr(policy, 'name', a.policy)}   "
                  f"(true reliability r = {ep.r:.2f})", fontsize=8)

    def draw(i):
        (x, y), searched, note = frames[i]
        robot.set_data([x], [y])
        for k, shade in enumerate(shades):
            shade.set_alpha(0.35 if searched[k] else 0.0)
        title.set_text(note)
        return [robot, title, *shades]

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=80, blit=False)
    anim.save(a.out, writer=PillowWriter(fps=12))
    print(f"wrote {a.out} ({len(frames)} frames, {'success' if search.success else 'failure'}, "
          f"{search.distance_m:.1f} m planned)")


if __name__ == "__main__":
    main()
