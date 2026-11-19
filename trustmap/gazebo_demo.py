"""Record the hand-picked Gazebo demo episodes of PPO-Conf (DECISIONS "Demo material"; demo only).

Runs under ROS 2 Humble's Python with the launch file running for one house
(gazebo/run_house.sh does both; see results/demo/README.md):

    PYTHONPATH=. python3 -m trustmap.gazebo_demo --house 200000 --selection results/demo/selection.json

For every selected episode of this house (trustmap.demo_select), this does what
gazebo_runner does for one episode (same object placement, same policy, Nav2,
one retry, virtual camera at the true pose), plus, for the demo only:
  - saves every frame of the fixed overhead camera (/overhead/image) with its
    simulation timestamp to results/demo/<house>_<episode>/frames/;
  - turns a full circle at every viewpoint (the 360-degree look of Q4);
  - when the target is detected, turns the robot toward the object and saves
    one picture from the robot's demo camera (/camera/image) as still.png;
  - writes episode.json: what the policy is shown at the start, the hidden
    values (r, the object's real room), and per decision the chosen room, the
    optimum's choice in the same state, that decision's regret in meters and
    the outcome, all with simulation timestamps so the video panel can be
    synchronised (trustmap.demo_video builds the MP4 from these files).
Nothing here feeds any reported result; the Gazebo result rows come only from
gazebo_runner.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from trustmap.camera import target_visible
from trustmap.demo_select import house_episodes
from trustmap.episode import DEFAULT_MAX_DECISIONS
from trustmap.gazebo_export import overhead_camera
from trustmap.gazebo_runner import Runner, load_policy, show_target, teleport
from trustmap.generator import generate_house, place_object
from trustmap.optimum import solve_episode
from trustmap.pool import load_pool

REPO = Path(__file__).resolve().parent.parent
POLICY = REPO / "gazebo" / "policies" / "ppo_conf.npz"


def image_to_rgb(msg):
    """sensor_msgs/Image (rgb8 or bgr8) to an RGB numpy array."""
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
    a = a.reshape(msg.height, msg.width, 3)
    return a[..., ::-1] if msg.encoding.lower().startswith("bgr") else a


def room_labels(room_types):
    """'bedroom 1', 'bedroom 2' when a type appears twice, otherwise the type name."""
    labels = []
    for k, t in enumerate(room_types):
        if list(room_types).count(t) > 1:
            labels.append(f"{t} {list(room_types[:k + 1]).count(t)}")
        else:
            labels.append(t)
    return labels


class DemoRunner(Runner):
    def __init__(self):
        super().__init__()
        import cv2
        from sensor_msgs.msg import Image

        self.cv2 = cv2
        self.frames_dir = None
        self.frames = 0
        self.camera = None  # latest demo-camera message
        self.node.create_subscription(Image, "/overhead/image", self._on_overhead, 10)
        self.node.create_subscription(Image, "/camera/image", self._on_camera, 10)

    def now(self):
        """Simulation time in seconds (use_sim_time), the clock of the image stamps."""
        return self.node.get_clock().now().nanoseconds / 1e9

    def _on_overhead(self, msg):
        if self.frames_dir is None:
            return
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        rgb = image_to_rgb(msg)
        self.cv2.imwrite(str(self.frames_dir / f"{stamp:016d}.jpg"), rgb[..., ::-1],
                         [self.cv2.IMWRITE_JPEG_QUALITY, 90])
        self.frames += 1

    def _on_camera(self, msg):
        self.camera = msg

    def turn_to(self, yaw, timeout_s=20.0):
        """Turn on the spot until the robot faces `yaw` (radians), with a simple P controller."""
        msg = self.Twist()
        for _ in range(int(timeout_s / 0.05)):
            err = math.atan2(math.sin(yaw - self.pose[2]), math.cos(yaw - self.pose[2]))
            if abs(err) < 0.03:
                break
            w = max(-0.8, min(0.8, 1.5 * err))
            msg.angular.z = math.copysign(max(abs(w), 0.15), w)
            self.cmd.publish(msg)
            self.spin(0.05)
        self.cmd.publish(self.Twist())
        self.spin(0.5)

    def save_still(self, path, after_s, timeout_s=5.0):
        """Save the first demo-camera picture taken after simulation time `after_s`."""
        for _ in range(int(timeout_s / 0.05)):
            m = self.camera
            if m is not None and m.header.stamp.sec + m.header.stamp.nanosec / 1e9 > after_s:
                self.cv2.imwrite(str(path), image_to_rgb(m)[..., ::-1])
                return True
            self.spin(0.05)
        return False


def record_episode(runner, layout, houses, item, policy, out_root):
    seed, i, rho = item["house"], item["episode"], item["rho"]
    house = layout.house
    ep = {k: e for k, _, e in house_episodes(seed, rho, houses)}[i]
    sol = solve_episode(house, ep, DEFAULT_MAX_DECISIONS)
    out = out_root / f"{seed}_{i}"
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for f in frames.glob("*.jpg"):
        f.unlink()

    # Same placement and random stream as gazebo_runner, so the episode is identical.
    rng = np.random.default_rng([seed, i])
    obj = place_object(layout, ep.object_room, rng)
    show_target(*obj)
    teleport(*layout.viewpoint_xy(ep.start_room))
    runner.spin(1.0)
    runner.clear_costmaps()
    runner.driven = 0.0
    runner.frames = 0
    runner.frames_dir = frames
    runner.spin(3.0)  # a few seconds of the start position for the video
    events = [{"t": runner.now(), "kind": "start"}]

    from trustmap.search import RoomSearch

    search = RoomSearch(house, ep, DEFAULT_MAX_DECISIONS)
    decisions, nav_failures, still = [], 0, None
    while not search.done:
        cur, bits = search.current_room, search.searched_bits
        q = sol.q_values(cur, bits)
        row = policy.choose(search, rng)
        room = int(ep.row_order[row])
        best = int(np.argmax(q))
        regret_m = float((q.max() - q[room]) * house.max_distance)
        d = {"k": len(decisions) + 1, "from_room": cur, "room": room, "optimum_room": best,
             "regret_m": round(regret_m, 3), "planned_m": round(float(house.distances[cur, room]), 2),
             "t_decide": runner.now()}
        gx, gy = layout.viewpoint_xy(room)
        ok = runner.navigate(gx, gy) or runner.navigate(gx, gy)  # one retry
        if not ok:
            nav_failures += 1
            teleport(gx, gy)
            runner.spin(1.0)
        d["t_arrive"] = runner.now()
        d["nav_ok"] = ok
        runner.turn_full_circle()
        seen = target_visible(layout.grid, runner.pose, obj, 2 * math.pi)
        search.step(row)
        d["outcome"] = "found" if search.success else ("failure" if search.done else "not here")
        d["camera_saw_target"] = bool(seen)
        d["t_outcome"] = runner.now()
        if search.success:
            yaw = math.atan2(obj[1] - runner.pose[1], obj[0] - runner.pose[0])
            runner.turn_to(yaw)
            t = runner.now()
            if runner.save_still(out / "still.png", t):
                still = "still.png"
            d["t_still"] = runner.now()
        decisions.append(d)
        print(f"  decision {d['k']}: {d}", flush=True)
    runner.spin(3.0)  # hold the end for the video
    events.append({"t": runner.now(), "kind": "end"})
    runner.frames_dir = None

    (cx, cy, cz), fov = overhead_camera(layout)
    cs = layout.grid.cell_size
    info = {
        "label": "hand-picked illustrative example; not part of the Gazebo result rows",
        "house": seed, "episode": i, "rho": rho, "kappa": 1.0, "policy": "PPO-Conf (median-validation seed)",
        "shown_to_robot": {
            "target": ep.target, "start_room": ep.start_room, "claimed_room": ep.claimed_room,
            "c": round(float(ep.c), 3), "habit": [round(float(h), 3) for h in ep.habit],
        },
        "hidden_from_robot": {"r": round(float(ep.r), 3), "object_room": ep.object_room,
                              "claim_correct": bool(ep.claim_correct)},
        "room_types": list(house.room_types), "room_labels": room_labels(house.room_types),
        "rooms_m": [[v * cs for v in r.rect] for r in layout.rooms],
        "object_xy_m": [float(obj[0]), float(obj[1])],
        "max_distance_m": float(house.max_distance),
        "decisions": decisions,
        "success": bool(search.success), "planned_m": round(search.distance_m, 2),
        "driven_m": round(runner.driven, 2),
        "regret_m": round(sum(d["regret_m"] for d in decisions), 3),
        "nav_failures": nav_failures, "still": still, "frames": runner.frames,
        "events": events,
        "overhead_camera": {"xyz_m": [cx, cy, cz], "horizontal_fov": fov},
    }
    (out / "episode.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {out} ({runner.frames} overhead frames, success={search.success})", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--house", type=int, required=True)
    p.add_argument("--selection", default=str(REPO / "results" / "demo" / "selection.json"))
    p.add_argument("--out", default=str(REPO / "results" / "demo"))
    a = p.parse_args()
    items = [e for e in json.loads(Path(a.selection).read_text())["episodes"] if e["house"] == a.house]
    if not items:
        raise SystemExit(f"no selected episodes for house {a.house}")
    layout = generate_house(a.house)
    houses = load_pool("test")
    policy = load_policy(str(POLICY))
    runner = DemoRunner()
    try:
        for item in items:
            print(f"episode {item['house']}_{item['episode']} (rho {item['rho']})", flush=True)
            record_episode(runner, layout, houses, item, policy, Path(a.out))
    finally:
        runner.frames_dir = None
        runner.shutdown()


if __name__ == "__main__":
    main()
