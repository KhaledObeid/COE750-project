# Should the Robot Trust Its Map?

COE750 reinforcement learning course project (LAU, Fall 2026), Khaled Obeid. A robot searches a five-room house for a target object. A semantic prior map claims one room with a confidence c; the map is wrong a controlled fraction of the time (corruption ρ) and its confidence may be miscalibrated (κ). We test whether a PPO policy learns the right amount of trust in the map, measured against an exact value-iteration optimum. Training runs in our own 2D floor-plan simulator; the frozen policy is tested in Gazebo Fortress with Nav2.

Every design decision and its reason is in `docs/DECISIONS.md`; current progress is in `docs/STATUS.md`.

## Install

The 2D simulator, training and experiments are pure Python on CPU and run on Ubuntu and Windows (Python 3.10 or newer). The versions used for the reported results are pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # Linux only: CPU build of PyTorch
pip install -r requirements.txt
pip install -e .
```

The Gazebo part needs Ubuntu 22.04 with ROS 2 Humble, Gazebo Fortress and `sudo apt install ros-humble-ros-gz ros-humble-navigation2 ros-humble-nav2-bringup`. It runs under ROS's own Python (`/usr/bin/python3`, 3.10), which needs only numpy: the trained policy is exported to plain numpy weights for it.

## Reproduce the results

Each command writes its results files under `results/` and redraws its figures into `results/figures/` (PDF for the report, PNG for slides). Commands with `--figures-only` redraw from the saved results without recomputing anything.

```bash
python -m trustmap.pool                      # 1. house pools: 5,000 train / 200 validation / 200 test (about 10 min on 8 cores)
python -m trustmap.plot_house 0 1 2 3        #    look at houses (seed 0 is the development house)
python -m trustmap.train_all --workers 7     # 2. all reported policies: 15 main runs + 3 at kappa 3 + 3 uniform-placement runs
python -m trustmap.e0                        # 3. E0: tabular Q-learning reaches the optimum      -> results/e0/
python -m trustmap.e1                        # 4. E1 + H1/H4: main comparison across rho          -> results/e1/
python -m trustmap.e2                        # 5. E2 + H2: follow rate against each episode's c*  -> results/e2/
python -m trustmap.e3                        # 6. E3 + H3: calibration 2 x 2                       -> results/e3/
python -m trustmap.e1 --condition lam0 lam1  #    lambda sensitivity row (evaluation only, Q10)    -> results/e1/table.csv
python -m trustmap.h4_check                  #    H4 check (post hoc): gap by habit x claimed room  -> results/e1/h4_check.csv
pip install prior==1.0.3                     #    ProcTHOR-10K row (no training; needs internet once to fetch the dataset):
python -m trustmap.procthor convert          #    100 five-room houses on our grid                   -> data/procthor/, results/procthor/houses.png
python -m trustmap.procthor evaluate         #    15 main policies + baselines                      -> results/procthor/table.csv
python -m trustmap.procthor_check           #    conversion check against ProcTHOR's own data     -> results/procthor/conversion_check.csv
```

What is in the repository and what is regenerated: every run folder under `runs/` (`best_model.zip`, `final_model.zip`, `config.json`, `validation_curve.csv`), the house pools in `data/houses/` (they are small: only room types and distance tables; the grids are rebuilt from the seed), and all results files and figures are committed. Not committed: the training logs `runs/*/train.log` (rerun `train_all`), the Gazebo worlds `gazebo/worlds/` (rerun `gazebo_export`), and the raw overhead frames of the demo (deleted after the videos are built).

One training run takes about 10 minutes alone, or about 30 minutes when 7 runs share 8 cores. A single run: `python -m trustmap.train --variant conf --seed 0` (variants `conf`, `binary`, `noprior`; options `--kappa`, `--placement uniform`). Each run folder under `runs/` holds `config.json`, the learning curve `validation_curve.csv` and `best_model.zip`, the checkpoint with the lowest validation regret, which is the one evaluated. A quick check of one policy against the baselines on 200 episodes per ρ: `python -m trustmap.evaluate --model runs/conf_kappa1_lam0.5_table_seed0/best_model.zip --variant conf`.

### Gazebo test (E-Gazebo, Point 9e)

```bash
python -m trustmap.gazebo_export 200000 200001 200002 200003                     # worlds, Nav2 maps, robot, overhead demo camera
python -m trustmap.policy_export --median conf --out gazebo/policies/ppo_conf.npz  # PPO-Conf seed with the median validation regret

# unattended, one house at a time: launches Gazebo + bridge + Nav2 headless, waits for Nav2, runs, shuts everything down
gazebo/run_house.sh 200000 results/gazebo -- --policy gazebo/policies/ppo_conf.npz --policy follower
gazebo/run_house.sh 200001 results/gazebo -- --policy gazebo/policies/ppo_conf.npz --policy follower
python -m trustmap.gazebo_summary                                                # both result rows -> results/gazebo/summary.csv

# or by hand, in two terminals:

source /opt/ros/humble/setup.bash
ros2 launch gazebo/launch/house.launch.py house:=200000                         # terminal 1 (add headless:=true for no window)
PYTHONPATH=$PWD:$PYTHONPATH /usr/bin/python3 -m trustmap.gazebo_runner --house 200000 \
    --policy gazebo/policies/ppo_conf.npz --policy follower                     # terminal 2 -> results/gazebo/
```

The runner plays the same 20 test episodes per house (ρ = 0.4) for both policies, retries a failed Nav2 goal once, and writes both result rows: decision metrics on planned distance, which must equal the 2D results on the same episodes (column `same_as_2d`), and end-to-end metrics on driven distance (`driven_m`, `overhead`, `nav_failures`). `--spin` makes the robot turn a full circle at every viewpoint; the demo camera streams on `/camera/image`.

### Demo videos (DECISIONS "Demo material"; illustration only, not a reported result)

```bash
python -m trustmap.demo_select                                   # pick about 10 PPO-Conf episodes by a fixed rule -> results/demo/selection.json
gazebo/run_house.sh 200000 results/demo/logs --module trustmap.gazebo_demo -- --selection results/demo/selection.json
#   ... the same for 200001, 200002, 200003
python -m trustmap.demo_video                                    # overhead video + side panel -> results/demo/<house>_<episode>/video.mp4
python -m trustmap.animate_episode --house 200000 --episode 8 --policy gazebo/policies/ppo_conf.npz --out results/demo/demo_2d.gif
```

`results/demo/README.md` lists the selected episodes, the selection rule and what each video shows.

## Run the tests

```bash
python -m pytest
```

On a machine with ROS 2 sourced, ROS's pytest plugins and Python path can break the run; use `PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest`. Tests that need the RL libraries or the built house pools are skipped when those are missing.

## Code layout

World and rules (shared by the 2D simulator and Gazebo):
- `trustmap/cooccurrence.py`: room types, targets, co-occurrence table (Point 3b)
- `trustmap/house.py`, `trustmap/generator.py`, `trustmap/pool.py`: house data, house generator (rooms, doors, grid, viewpoints, distances, object placement) and the saved pools (Point 3a)
- `trustmap/grid.py`, `trustmap/planning.py`, `trustmap/camera.py`: occupancy grid, A* and the distance table, virtual camera
- `trustmap/episode.py`: episode generation recipe (Point 3c)
- `trustmap/observation.py`: observation rows and comparison-variant masks (Point 4)
- `trustmap/search.py`: room-level search rules: action mask, episode end, reward (Points 5, 6)

Optimum, metrics and baselines:
- `trustmap/optimum.py`: exact optimum by value iteration (160 states), fixed-order return lines and the trust threshold c*
- `trustmap/metrics.py`: success, SPL, first-decision success, decision regret (Points 6b, 8b)
- `trustmap/baselines.py`: random, nearest-unexplored, greedy map follower, Bayes-nominal (Point 8d)
- `trustmap/qlearning.py`: tabular Q-learning (Point 8c)

Learning and experiments:
- `trustmap/env.py`, `trustmap/ppo.py`, `trustmap/train.py`, `trustmap/train_all.py`: Gymnasium environment with action masks, MaskablePPO settings, training (Point 7)
- `trustmap/evaluation.py`, `trustmap/experiments.py`: paired evaluation on the fixed test set, saved records, confidence intervals (Point 9b)
- `trustmap/e0.py` to `trustmap/e3.py`: one command per experiment (Points 8c, 9a-9d, 10a); `trustmap/evaluate.py`: quick check

Gazebo:
- `trustmap/gazebo_export.py`: SDF world, Nav2 map and robot from a generated house
- `trustmap/policy_export.py`: numpy copy of a trained policy for ROS's Python
- `trustmap/gazebo_runner.py`, `gazebo/launch/house.launch.py`, `gazebo/config/`, `gazebo/run_house.sh`: runner, launch file, bridge and Nav2 settings, unattended run of one house
- `trustmap/gazebo_summary.py`: the two Gazebo result rows from the runner's per-episode files
- `trustmap/demo_select.py`, `trustmap/gazebo_demo.py`, `trustmap/demo_video.py`: demo episode selection, recording in Gazebo, video with side panel

Other: `trustmap/animate_episode.py` (2D episode animation, the backup demo: `python -m trustmap.animate_episode --house 200000 --episode 1 --policy follower --out demo.gif`), `trustmap/plot_house.py` (house plots), `trustmap/smoke_houses.py` (stand-in houses for debugging only), `report/` (IEEE LaTeX report), `tests/`.

## License and acknowledgements

MIT license (`LICENSE`). This project uses Stable-Baselines3 and sb3-contrib, Gymnasium, PyTorch, ROS 2, Nav2 and Gazebo. Claude (Anthropic) was used as a design-discussion partner and coding assistant; the author is responsible for every choice and result.
