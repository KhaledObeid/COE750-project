# Should the Robot Trust Its Map?

COE750 reinforcement learning course project (LAU, Fall 2026). A robot searches a house for an object using a semantic prior map whose claims are wrong a controlled fraction of the time and whose confidence may be miscalibrated; we test whether a PPO policy learns the right amount of trust, compared against an exact optimum.

Work in progress. Project decisions are in `docs/DECISIONS.md`, current progress in `docs/STATUS.md`.

## Install

Pure Python on CPU; works on Linux and Windows. Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

For training (Point 7) also install the RL extra. On Linux, install the CPU build of PyTorch first so pip does not download the large GPU build; on Windows the default build is already CPU-only.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux only
pip install -e ".[dev,rl,plot]"
```

## Train and evaluate

```bash
python -m trustmap.pool                                   # once: builds the house pools into data/houses/ (about 10 min on 8 cores)
python -m trustmap.plot_house 0 1 2 --out houses.png      # check houses by eye
python -m trustmap.train --variant conf --seed 0          # variants: conf, binary, noprior
python -m trustmap.evaluate --model runs/conf_kappa1_lam0.5_table_seed0/best_model.zip --variant conf
```

Each run writes `config.json`, `validation_curve.csv` (the learning curve), `best_model.zip` (lowest validation regret) and `final_model.zip` under `runs/`. `--houses smoke` switches both commands to stand-in houses (`trustmap/smoke_houses.py`), for debugging only.

## Run the tests

```bash
python -m pytest
```

On a machine with ROS 2 sourced, ROS's pytest plugins and Python path can break the run; use `PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest` (the same `PYTHONPATH=` prefix helps for training). Tests for the environment are skipped when the RL extra is not installed.

## Code layout

- `trustmap/cooccurrence.py`: room types, targets, co-occurrence table (Point 3b)
- `trustmap/house.py`: house data and room-type draw (Points 3a, 3b)
- `trustmap/episode.py`: episode generation recipe (Point 3c)
- `trustmap/observation.py`: observation rows and comparison-variant masks (Point 4)
- `trustmap/search.py`: room-level search rules: action mask, episode end, reward (Points 5, 6)
- `trustmap/optimum.py`: exact optimum by value iteration over current room × searched set
- `trustmap/metrics.py`: success, SPL, success on the first decision (Point 6b), decision regret (Point 8b)
- `trustmap/grid.py`: occupancy grid and obstacle inflation, shared by the 2D simulator and Gazebo
- `trustmap/planning.py`: A* and the room-to-room planned-distance table (Point 3a)
- `trustmap/camera.py`: virtual camera, view cone plus line-of-sight ray casting
- `trustmap/baselines.py`: random, nearest-unexplored, greedy map follower, Bayes-nominal (Point 8d)
- `trustmap/qlearning.py`: tabular Q-learning for the E0 sanity check (Point 8c)
- `trustmap/env.py`: Gymnasium environment with `action_masks()` (Point 7)
- `trustmap/ppo.py`: MaskablePPO settings and network (Points 7b, 7c)
- `trustmap/train.py`: training with validation-regret checkpoints (Point 7d)
- `trustmap/evaluation.py`, `trustmap/evaluate.py`: paired evaluation of all methods on fixed test episodes
- `trustmap/generator.py`: house generator: room layout, doors, occupancy grid, viewpoints, distances, object placement (Point 3a)
- `trustmap/pool.py`: builds and loads the training, validation and test house pools
- `trustmap/plot_house.py`: plots houses for checking by eye
- `trustmap/smoke_houses.py`: stand-in houses with straight-line distances, for debugging only
- `tests/`: checks against the worked examples in `docs/DECISIONS.md`
