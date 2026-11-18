"""Run frozen policies in Gazebo with Nav2 and record both result rows (Point 9e).

Runs under ROS 2 Humble's Python (3.10) with the launch file already running:

    source /opt/ros/humble/setup.bash
    ros2 launch gazebo/launch/house.launch.py house:=200000 headless:=true      # terminal 1
    PYTHONPATH=. python3 -m trustmap.gazebo_runner --house 200000 \\
        --policy gazebo/policies/ppo_conf.npz --policy follower                  # terminal 2

Episodes (9e): the first 20 E1 test episodes of this house at rho = 0.4 (the
same test set as the 2D results, 9b), identical for every policy. Per episode:
  1. the target is placed on a random valid cell of its room (Q4) and shown as
     a visual-only model; the robot is moved to the start room's viewpoint;
  2. at every decision the policy chooses a room through the shared RoomSearch
     and observation code, so its decisions are exactly those of the 2D
     simulator; Nav2 drives to the room's viewpoint;
  3. on arrival the virtual camera (cone + line of sight on the shared grid)
     is checked at the robot's true pose; with --spin the robot first turns a
     full circle on the spot for the demo video.
If Nav2 fails a goal it is retried once; if it fails again a navigation
failure is logged (end-to-end row only) and the robot is moved to the
viewpoint so the decision episode can finish ("Cost of a decision" lock).

Two rows (results/gazebo/): decision metrics on planned distance (must equal
the 2D results on the same episodes; checked here), and end-to-end metrics on
driven distance (driving overhead = driven / planned - 1, Nav2 failure rate,
end-to-end success). Gazebo results never feed back into training.
"""

import argparse
import csv
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from trustmap.baselines import GreedyMapFollower, NearestUnexplored
from trustmap.camera import target_visible
from trustmap.episode import DEFAULT_MAX_DECISIONS, EpisodeConfig
from trustmap.evaluation import make_test_episodes, play
from trustmap.gazebo_export import ROBOT_NAME, WORLD_NAME, target_sdf
from trustmap.generator import generate_house, place_object
from trustmap.metrics import episode_metrics
from trustmap.optimum import solve_episode
from trustmap.pool import load_pool
from trustmap.search import RoomSearch

RHO = 0.4                  # 9e
EPISODES_PER_HOUSE = 20    # 9e
TEST_SEED = 200_000        # 9b: the E1 test set
GOAL_TIMEOUT_S = 120.0
REPO = Path(__file__).resolve().parent.parent


def gazebo_episodes(seed, count=EPISODES_PER_HOUSE):
    """The first `count` E1 test episodes of test house `seed` at rho = 0.4."""
    houses = load_pool("test")
    index = [h.seed for h in houses].index(seed)
    cfg = EpisodeConfig(rho=RHO, kappa=1.0)
    all_eps = make_test_episodes(houses, cfg, 25, TEST_SEED)
    return [ep for _, ep in all_eps[index * 25: index * 25 + count]]


def gz_service(service, reqtype, reptype, req):
    """Call a Gazebo service through the `ign service` command line."""
    out = subprocess.run(["ign", "service", "-s", f"/world/{WORLD_NAME}/{service}", "--reqtype", reqtype,
                          "--reptype", reptype, "--timeout", "5000", "--req", req],
                         capture_output=True, text=True)
    return "true" in out.stdout


def teleport(x, y, yaw=0.0):
    qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
    return gz_service("set_pose", "ignition.msgs.Pose", "ignition.msgs.Boolean",
                      f'name: "{ROBOT_NAME}", position: {{x: {x}, y: {y}, z: 0.0}}, '
                      f"orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}")


def show_target(x, y):
    gz_service("remove", "ignition.msgs.Entity", "ignition.msgs.Boolean", 'name: "target", type: MODEL')
    sdf = target_sdf("target", x, y).replace('"', '\\"')
    return gz_service("create", "ignition.msgs.EntityFactory", "ignition.msgs.Boolean", f'sdf: "{sdf}"')


def load_policy(spec):
    if spec == "follower":
        return GreedyMapFollower()
    if spec == "nearest":
        return NearestUnexplored()
    from trustmap.policy_export import NumpyPolicy

    return NumpyPolicy(spec, name=f"{Path(spec).stem}")


class Runner:
    def __init__(self):
        import rclpy
        from geometry_msgs.msg import Twist
        from nav2_msgs.action import NavigateToPose
        from nav2_msgs.srv import ClearEntireCostmap
        from nav_msgs.msg import Odometry
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from rclpy.parameter import Parameter

        rclpy.init()
        self.rclpy = rclpy
        self.node = Node("trustmap_runner", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.NavigateToPose = NavigateToPose
        self.Twist = Twist
        self.nav = ActionClient(self.node, NavigateToPose, "navigate_to_pose")
        self.cmd = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.clear = [self.node.create_client(ClearEntireCostmap, s) for s in
                      ("/global_costmap/clear_entirely_global_costmap", "/local_costmap/clear_entirely_local_costmap")]
        self.ClearRequest = ClearEntireCostmap.Request
        self.pose = None
        self.driving = False
        self.driven = 0.0
        self.node.create_subscription(Odometry, "/odom", self._on_odom, 50)
        self.node.get_logger().info("waiting for Nav2 and the ground-truth pose ...")
        self.nav.wait_for_server()
        while self.pose is None:
            self.spin(0.1)

    def _on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        if self.pose is not None and self.driving:
            self.driven += math.hypot(p.x - self.pose[0], p.y - self.pose[1])
        self.pose = (p.x, p.y, yaw)

    def spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def clear_costmaps(self):
        for c in self.clear:
            if c.wait_for_service(timeout_sec=2.0):
                self.rclpy.spin_until_future_complete(self.node, c.call_async(self.ClearRequest()), timeout_sec=5.0)

    def navigate(self, x, y):
        """Drive to (x, y) with Nav2; True if Nav2 reports success."""
        from action_msgs.msg import GoalStatus

        goal = self.NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.w = 1.0
        self.driving = True
        try:
            sent = self.nav.send_goal_async(goal)
            self.rclpy.spin_until_future_complete(self.node, sent, timeout_sec=10.0)
            handle = sent.result()
            if handle is None or not handle.accepted:
                return False
            result = handle.get_result_async()
            self.rclpy.spin_until_future_complete(self.node, result, timeout_sec=GOAL_TIMEOUT_S)
            if not result.done():
                handle.cancel_goal_async()
                return False
            return result.result().status == GoalStatus.STATUS_SUCCEEDED
        finally:
            self.driving = False

    def turn_full_circle(self, speed=0.8):
        """Demo only: turn on the spot once (the 360-degree look of Q4)."""
        msg = self.Twist()
        msg.angular.z = speed
        end = time.time() + 2 * math.pi / speed
        while time.time() < end:
            self.cmd.publish(msg)
            self.spin(0.05)
        self.cmd.publish(self.Twist())

    def shutdown(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def run(house_seed, policy_specs, count, spin_demo, out_dir):
    layout = generate_house(house_seed)
    house = layout.house
    episodes = gazebo_episodes(house_seed, count)
    runner = Runner()
    rows = []
    try:
        for spec in policy_specs:
            policy = load_policy(spec)
            for i, ep in enumerate(episodes):
                rng = np.random.default_rng([house_seed, i])
                obj = place_object(layout, ep.object_room, rng)
                show_target(*obj)
                teleport(*layout.viewpoint_xy(ep.start_room))
                runner.spin(1.0)
                runner.clear_costmaps()
                runner.driven = 0.0

                search = RoomSearch(house, ep, DEFAULT_MAX_DECISIONS)
                nav_failures = 0
                camera_agrees = True
                camera_found = False
                while not search.done:
                    row = policy.choose(search, rng)
                    room = int(ep.row_order[row])
                    gx, gy = layout.viewpoint_xy(room)
                    ok = runner.navigate(gx, gy) or runner.navigate(gx, gy)  # one retry
                    if not ok:
                        nav_failures += 1
                        teleport(gx, gy)  # let the decision episode finish (end-to-end row records the failure)
                        runner.spin(1.0)
                    if spin_demo:
                        runner.turn_full_circle()
                    seen = target_visible(layout.grid, runner.pose, obj, 2 * math.pi)
                    camera_found = camera_found or seen
                    camera_agrees = camera_agrees and (seen == (room == ep.object_room))
                    search.step(row)

                m = episode_metrics(search, solve_episode(house, ep, DEFAULT_MAX_DECISIONS))
                twin = play(load_policy(spec), house, ep, DEFAULT_MAX_DECISIONS, np.random.default_rng([house_seed, i]))
                rows.append({
                    "house": house_seed, "episode": i, "policy": getattr(policy, "name", spec),
                    "success": int(m.success), "spl": round(m.spl, 4), "regret_m": round(m.regret_m, 4),
                    "success_first_decision": int(m.success_first_decision), "decisions": m.decisions,
                    "planned_m": round(m.distance_m, 3), "driven_m": round(runner.driven, 3),
                    "overhead": round(runner.driven / m.distance_m - 1, 4) if m.distance_m > 0 else 0.0,
                    "nav_failures": nav_failures,
                    "end_to_end_success": int(m.success and nav_failures == 0 and camera_found),
                    "camera_agrees": int(camera_agrees),
                    "same_as_2d": int([h[2] for h in twin.history] == [h[2] for h in search.history]),
                })
                print(rows[-1], flush=True)
    finally:
        runner.shutdown()
        out_dir.mkdir(parents=True, exist_ok=True)
        if rows:
            path = out_dir / f"house_{house_seed}.csv"
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--house", type=int, required=True)
    p.add_argument("--policy", action="append", required=True,
                   help="'follower', 'nearest', or an exported policy .npz (repeat for several)")
    p.add_argument("--episodes", type=int, default=EPISODES_PER_HOUSE)
    p.add_argument("--spin", action="store_true", help="turn a full circle at every viewpoint (demo videos)")
    p.add_argument("--out", default=str(REPO / "results" / "gazebo"))
    a = p.parse_args()
    run(a.house, a.policy, a.episodes, a.spin, Path(a.out))


if __name__ == "__main__":
    main()
