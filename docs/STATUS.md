# STATUS

Written by Claude Code only. Updated at the end of every coding session.

## Current state (2026-11-19)

All eleven points are locked and every one now has code. In the `trustmap` package:

- World and rules: `cooccurrence.py`, `house.py`, `generator.py`, `pool.py` (3a, 3b, Q2–Q5, Q9), `grid.py`, `planning.py`, `camera.py` (shared with Gazebo), `episode.py` (3c), `observation.py` (4), `search.py` (5, 6).
- Optimum and checks: `optimum.py` (value iteration; fixed-order return lines; per-episode trust threshold c*, 9c), `metrics.py` (6b, 8b regret), `baselines.py` (8d), `qlearning.py` (8c).
- Learning: `env.py`, `ppo.py`, `train.py`, `train_all.py` (7, 9a; all locked runs in one command, 7 in parallel).
- Experiments (9, 10a): `evaluation.py` (9b test set: 25 episodes per test house, 5,000 per ρ, seed 200,000), `experiments.py` (shared records, 95% t-intervals over seeds), `e0.py`, `e1.py` (H1, H4), `e2.py` (H2), `e3.py` (H3). Each writes `results/<exp>/` and `results/figures/` and can redraw with `--figures-only`.
- Gazebo (9e, Q9.7): `gazebo_export.py` (SDF world from the exact grid cells, floor colours, no furniture, robot with casters, lidar and demo camera, Nav2 map), `policy_export.py` (numpy copy of a policy for ROS's Python 3.10; median-validation seed), `gazebo_runner.py` (both result rows), `gazebo/launch/house.launch.py`, `gazebo/config/` (bridge, Nav2 params).
- Packaging (11): README with one command per result, pinned `requirements.txt`, MIT `LICENSE`, `report/` (IEEEtran skeleton with every figure wired in, `refs.bib` stub to be checked), `animate_episode.py` (2D backup demo).

Tested: 63 pytest tests pass in `.venv` (Python 3.14). New checks this session: the best fixed-order line equals V* and c* is exactly where the optimum's first choice switches (300 episodes); on the E1 machinery the value-iteration optimum follows the claim exactly when c ≥ c* (0 mismatches in 923 episodes with a c*); the numpy policy picks the same room as SB3 in every decision of 500 episodes; experiment records and t-intervals. The core modules also run under ROS's Python 3.10 / numpy 1.21 and rebuild identical houses. Gazebo: both exported worlds pass `ign sdf -k`; a headless run (Gazebo + bridge + Nav2) completed 2 episodes in house 200,000 with the follower: Nav2 reached every goal, the virtual camera at the true pose agreed with the search rules, decisions equal to 2D, driven distance within 4% of planned.

E0 (results committed): after 50,000 Q-learning episodes on 20 configurations, |Q − Q*| at the start is 0.003 house-widths, the first room matches the optimum in 19/20 (the 20th is a near-tie: 0.0013 house-widths between the two best rooms), and 98% of the states on the optimum's path match; over all decision states 82% match, since states reached only after an unusual choice are rarely visited.

Training: see the results paragraph below.

Open (OPEN_QUESTIONS): Q10 (λ row and uniform row: what is trained), Q11 (Gazebo choices: ground-truth pose instead of AMCL, test houses 200,000/200,001, Nav2 settings), Q12 (small E1/E2 analysis choices).

Not done: the 80-episode Gazebo run (2–3 h unattended, after Q11) and demo videos; the ProcTHOR-10K row (optional); E4 (stretch); writing the report text and the slides.

## Next coding step

Khaled answers Q10–Q12. Then: `python -m trustmap.policy_export --median conf --out gazebo/policies/ppo_conf.npz`, the Gazebo run for both houses (README), and, if Q10 asks for it, the λ row (`python -m trustmap.e1 --condition lam0 lam1`).

## Log

- 2026-11-17: repository set up; Points 3b, 3c, 4, 5, 6, exact optimum and metrics implemented.
- 2026-11-18: shared geometry; Points 7–8 (env, training, regret, baselines, E0 test); house generator and pools; Q7–Q9 added.
- 2026-11-19: Points 9–11: c*, E0–E3 scripts and figures, all locked runs trained, Gazebo export/launch/runner tested headless, README/requirements/license/report skeleton, 2D animation; Q10–Q12 added.
