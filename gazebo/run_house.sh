#!/usr/bin/env bash
# Run one Gazebo house unattended (Point 9e): launch Gazebo + bridge + Nav2 headless,
# wait until Nav2 is active, run the runner, then shut every process down.
#
#   gazebo/run_house.sh <house> <log_dir> [--bag <bag_dir>] [--module <python module>] -- <arguments>
#
# Examples (from the repository root):
#   gazebo/run_house.sh 200000 results/gazebo -- --policy gazebo/policies/ppo_conf.npz --policy follower
#   gazebo/run_house.sh 200000 results/demo/logs --module trustmap.gazebo_demo -- \
#       --selection results/demo/selection.json          # demo videos (trustmap.gazebo_demo)
#
# Logs: <log_dir>/launch_<house>.log (Gazebo, bridge, Nav2) and <log_dir>/runner_<house>.log.
# With --bag, /camera/image is recorded with ros2 bag while the runner drives.
# --module runs another module than trustmap.gazebo_runner (default) with the same arguments.
set -u
HOUSE=$1; LOG_DIR=$2; shift 2
BAG=""
MODULE=trustmap.gazebo_runner
if [ "${1:-}" = "--bag" ]; then BAG=$2; shift 2; fi
if [ "${1:-}" = "--module" ]; then MODULE=$2; shift 2; fi
[ "${1:-}" = "--" ] && shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p "$LOG_DIR"
set +u; source /opt/ros/humble/setup.bash; set -u

LAUNCH_LOG="$LOG_DIR/launch_$HOUSE.log"
RUNNER_LOG="$LOG_DIR/runner_$HOUSE.log"

cleanup() {
    # Stop the bag, then the whole launch process group, then anything left over.
    [ -n "${BAG_PID:-}" ] && kill -INT "$BAG_PID" 2>/dev/null && sleep 3
    [ -n "${LAUNCH_PID:-}" ] && kill -INT -- -"$LAUNCH_PID" 2>/dev/null
    sleep 8
    [ -n "${LAUNCH_PID:-}" ] && kill -KILL -- -"$LAUNCH_PID" 2>/dev/null
    pkill -KILL -f "ign gazebo" 2>/dev/null
    pkill -KILL -f "gz sim" 2>/dev/null
    pkill -KILL -f "parameter_bridge" 2>/dev/null
    pkill -KILL -f "/opt/ros/humble/lib/nav2_" 2>/dev/null
    pkill -KILL -f "static_transform_publisher" 2>/dev/null
    pkill -KILL -f "ros2 bag record" 2>/dev/null
    sleep 2
}
trap cleanup EXIT

# setsid gives the launch its own process group, so one kill reaches every child.
setsid ros2 launch gazebo/launch/house.launch.py house:="$HOUSE" headless:=true > "$LAUNCH_LOG" 2>&1 &
LAUNCH_PID=$!

# Both lifecycle managers (map server, navigation) print this line once their nodes are active.
for i in $(seq 1 180); do
    n=$(grep -c "Managed nodes are active" "$LAUNCH_LOG" 2>/dev/null || true)
    [ "${n:-0}" -ge 2 ] && break
    sleep 2
done
if [ "${n:-0}" -lt 2 ]; then
    echo "Nav2 did not become active within 6 minutes; see $LAUNCH_LOG" | tee "$RUNNER_LOG"
    exit 1
fi
echo "Nav2 active after about $((i * 2)) s"
sleep 5

if [ -n "$BAG" ]; then
    rm -rf "$BAG"
    ros2 bag record -o "$BAG" /camera/image > "$LOG_DIR/bag_$HOUSE.log" 2>&1 &
    BAG_PID=$!
    sleep 3
fi

PYTHONPATH="$REPO:${PYTHONPATH:-}" /usr/bin/python3 -m "$MODULE" --house "$HOUSE" "$@" \
    > "$RUNNER_LOG" 2>&1
CODE=$?
echo "runner exit code $CODE (log: $RUNNER_LOG)"
exit $CODE
