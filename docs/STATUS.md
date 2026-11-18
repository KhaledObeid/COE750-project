# STATUS

Written by Claude Code only. Updated at the end of every coding session.

## Current state (2026-11-18)

Every locked point (1–8 and the engineering parameters) is implemented. Built, in the `trustmap` package:

- World (3a, 3b, Q2–Q5): `cooccurrence.py` (room types, eight targets, co-occurrence table, `placement = uniform`), `house.py` (`House`: room types and 5×5 planned distances; fifth-room draw; pool seed ranges), `generator.py` (rectangular 8–12 m footprint, public/private zones, rooms by growth weight, 2 m minimum, door tree with 1.0 m centred doors and one bathroom door, 0.10 m grid with 0.20 m walls, 0.20 m inflation, viewpoints, 8-connected distance table, object placement visible only from its own room's viewpoint), `pool.py` (builds and loads `data/houses/{train,validation,test}.npz`), `plot_house.py`.
- Shared geometry for 2D and Gazebo: `grid.py` (occupancy grid, inflation), `planning.py` (A*, Dijkstra, distance table), `camera.py` (view cone, 1.5 m range, line-of-sight ray cast).
- Episodes and rules (3c, 4, 5, 6): `episode.py` (recipe steps 2–6, row shuffle, start-room check), `observation.py` (5×5 rows, variant masks), `search.py` (`RoomSearch`: action mask, success on detection, reward, 3-decision cap, −1 failure, γ = 1, decision history).
- Optimum and metrics (8a–8c): `optimum.py` (value iteration over 160 states, true r), `metrics.py` (success, SPL, first-decision success, expected decision regret in meters), `qlearning.py` (tabular Q-learning for E0).
- Baselines (8d): `baselines.py` (random, nearest-unexplored, greedy map follower, Bayes-nominal).
- Learning (7): `env.py` (Gymnasium env with `action_masks()`, flat 25-number observation), `ppo.py` (7b network, 7c settings, `PPOPolicy` wrapper), `train.py` (MaskablePPO, 8 environment copies, 1M decisions, validation regret every 50k, best checkpoint, learning curve CSV), `evaluation.py` / `evaluate.py` (paired evaluation of all methods per ρ).
- `smoke_houses.py`: stand-in houses with straight-line distances, for debugging only.

Pools built: 5,000 training houses (9.3 min on 8 cores), 200 validation, 200 test; fifth room bedroom 2,508 / office 1,276 / dining 1,216 in training; largest room-to-room distance median 14.6 m (range 9.3–18.8 m). Houses 0–5 were plotted and look as intended; seed 0 (11.7 × 10.4 m, kitchen, living, bathroom, bedroom, office) is the development house, pending Khaled's look.

Tested with `pytest`: 59 tests pass in the project `.venv` (Python 3.14, torch 2.15 CPU, SB3/sb3-contrib 2.9.2, Gymnasium 1.4); without the RL extra, 54 pass and the env tests are skipped. Coverage: the DECISIONS.md worked numbers (3b/3c), observation and masks, search rules, metrics and regret, value iteration equal to brute force over all 24 search orders on 2,000 random episodes (8c), Q-learning reaching the optimum's first room and Q-value (E0), baselines obeying mask and cap, Bayes-nominal equal to the optimum at κ = 1, the Gymnasium env checker, generator rules (sizes, zones, door tree, one bathroom door, door widths 1.0 m raw and 0.6 m after inflation, viewpoints, distances), and object detection only from the object's own room.

Timing (7d): a full 1M-decision PPO-Conf run took 9.8 min including 21 validation checks (on stand-in houses; the per-decision cost does not depend on the houses), below the 20-minute line, so 5 seeds per main variant stands.

Placeholders, not decided (OPEN_QUESTIONS Q7): training ρ ~ U[0, 0.8] per episode; 1,000 validation episodes at seed 100,000; greedy actions at evaluation; 200 test episodes per ρ at seed 200,000. Bayes-nominal falls back to the nearest room in the practically impossible case c = 1 with a wrong claim.

Not built: the Gazebo side (SDF export, robot, Nav2 runner), the ProcTHOR-10K converter, the E1–E4 experiment scripts and plots (Point 9 not locked).

## Next coding step

Khaled checks the seed-0 plot and answers Q7–Q9 in the RL chat. Then the 20 November go/no-go run: `python -m trustmap.train --variant conf --seed 0`, then `python -m trustmap.evaluate --model runs/conf_kappa1_lam0.5_table_seed0/best_model.zip --variant conf`. After Point 9 is locked: the experiment scripts (E0–E3), then the Gazebo export.

## Log

- 2026-11-17: repository set up from the claude.ai RL Project chat.
- 2026-11-17: Points 3b, 3c and 4 implemented and tested; generator questions logged.
- 2026-11-17: Points 5 and 6 (masking, episode end, reward, cap), exact optimum and 6b metrics implemented and tested; Q1 resolved, Q6 (regret definition) added.
- 2026-11-18: shared geometry (grid, A*, camera); Points 7–8 (env, MaskablePPO training, regret, baselines, E0, brute-force check); house generator and pools (Q2–Q5); Q7–Q9 added.
