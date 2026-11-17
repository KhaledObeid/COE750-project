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

## Run the tests

```bash
python -m pytest
```

On a machine with ROS 2 sourced, ROS's pytest plugins can break the run; disable them with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest`.

## Code layout

- `trustmap/cooccurrence.py`: room types, targets, co-occurrence table (Point 3b)
- `trustmap/house.py`: house data and room-type draw (Points 3a, 3b)
- `trustmap/episode.py`: episode generation recipe (Point 3c)
- `trustmap/observation.py`: observation rows and comparison-variant masks (Point 4)
- `tests/`: checks against the worked examples in `docs/DECISIONS.md`
