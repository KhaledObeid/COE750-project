# OPEN QUESTIONS

Written by Claude Code only. Questions the code raised that `docs/DECISIONS.md` does not answer. Khaled brings them to the claude.ai RL Project chat; once decided there, the answer goes into `DECISIONS.md` and the entry here is marked resolved with the date.

Format for each entry: date, file and function affected, the question, the options, Claude Code's recommendation, status (open or resolved on date).

## Open

### Q13 (2026-11-19): H4 check, ties for "the target's most likely room by habit"

- Affects: `trustmap/h4_check.py` `claim_on_habit_room`.
- Question: a house can have two rooms with the same highest habit weight (for example two bedrooms for a bedroom target). Does a claim on either of them count as "the claimed room was the most likely room by habit"? The start room (already searched) is also included when taking the maximum, as written ("in that house").
- Options: (a) any tied room counts; (b) only the lowest-index tied room counts; (c) exclude episodes with a tie.
- Decided provisionally by Claude Code, Khaled to confirm: (a), the simplest reading. Status: open.

## Resolved

### Q10 (2026-11-19): what the lambda sensitivity row and the uniform ablation row train

- Affects: `trustmap/train_all.py` (`GROUPS`), `trustmap/e1.py` (`CONDITIONS`, `methods_for`).
- Question: 7d lists "3 seeds for each extra row (PPO-Conf at kappa = 3, the uniform-placement ablation, the lambda sensitivity row), about 21 runs". 15 + 3 + 3 + 3 is 24, while 15 + 3 + 3 is exactly 21, which suggests the lambda row trains nothing. Two things are unclear. (a) The lambda row (3c: "an E1 sensitivity row varying lambda, for example 0, 0.5, 1"): either PPO-Conf is trained at another lambda (which one, or both 0 and 1, i.e. 3 or 6 runs), or the main policies trained at lambda = 0.5 are only tested at lambda = 0 and 1 (no training; this tests robustness to a changed error model rather than the conclusion in a different world). (b) The uniform ablation (3b): only PPO-Conf trained and tested with `placement = uniform` (3 runs), or also PPO-NoPrior so that the Conf minus NoPrior gap can be shown without commonsense.
- Implemented now (placeholders): `train_all` trains the uniform row as PPO-Conf only (3 seeds) and has no lambda group; `python -m trustmap.e1 --condition lam0 lam1` tests the main policies at lambda 0 and 1 (evaluation only), and is not run by default.
- Recommendation: (a) evaluation-only at lambda 0 and 1 for every main variant, which matches "about 21 runs" and needs no training; (b) PPO-Conf only, as implemented.
- Status: resolved 2026-11-19 (DECISIONS.md, answers to Q10-Q12): (a) the lambda row is evaluation-only, the main policies of every variant tested at lambda 0 and 1; (b) the uniform row trains PPO-Conf only (3 seeds). 21 runs in total. Code already matched; comments updated.

### Q11 (2026-11-19): Gazebo choices the 9e lock does not cover

- Affects: `trustmap/gazebo_export.py`, `gazebo/launch/house.launch.py`, `gazebo/config/nav2_params.yaml`, `trustmap/gazebo_runner.py`.
- Choices made, all working in a headless test (2 episodes in house 200,000: Nav2 reached every goal, the virtual camera agreed with the search rules, decisions identical to 2D, driven distance within 4% of planned):
  1. Localization: the robot pose comes from Gazebo's OdometryPublisher (ground truth); no AMCL. Nav2 still plans on the map and avoids obstacles with the lidar. Reason: each episode moves the robot to a new start room, which wheel odometry cannot follow, and AMCL would add a localization error that is not part of the trust question. The end-to-end row therefore measures driving (planning and control), not localization.
  2. Which two test houses: test seeds 200,000 and 200,001 (the first two of the test pool), unless Khaled prefers to pick two by eye.
  3. Nav2 settings: nav2_bringup's Humble defaults with robot radius 0.20 m, global costmap at 0.10 m, inflation radius 0.45 m, goal tolerance 0.10 m, max speed 0.30 m/s; per-goal timeout 120 s. Walls are 1.0 m high.
  4. When Nav2 fails a goal twice, the robot is moved to the viewpoint so the decision episode can finish; the failure counts only in the end-to-end row ("Cost of a decision").
  5. End-to-end success = decision success, no navigation failure, and the virtual camera saw the target at the robot's true pose.
- Recommendation: accept 1-5.
- Status: resolved 2026-11-19 (DECISIONS.md, answers to Q10-Q12): 1-5 accepted as implemented (ground-truth pose, no AMCL; houses 200,000 and 200,001; the listed Nav2 settings and 120 s goal timeout; move to the viewpoint after two failed attempts; end-to-end success definition).

### Q12 (2026-11-19): small analysis choices in E1 and E2

- Affects: `trustmap/e1.py`, `trustmap/e2.py`, `trustmap/optimum.py` `confidence_threshold`.
- Choices made: (1) E2 pools the E1 test episodes of all five rho values (25,000 at kappa = 1); (2) bins of c - c* are 0.05 wide over [-1, 1], and bins with fewer than 30 episodes are not drawn; (3) panel (b) splits PPO-Conf's episodes at the median scaled distance from the start room to the claimed room (near / far) to show the jump does not depend on distance; (4) H4 pairs PPO-NoPrior seed k with PPO-Conf seed k and gives a t-interval over the 5 paired differences; (5) episodes where the claim only ties at r = 1 (a room lies exactly on a shortest path to the claimed room, so searching it first is free) count as "no c*", like episodes where the optimum never switches; (6) at rho = 0, ties between the claimed room and such an on-the-way room are broken by room index in value iteration, so the optimum's and Bayes-nominal's first-decision success at rho = 0 can be slightly below 1 with zero regret.
- Recommendation: accept 1-6.
- Status: resolved 2026-11-19 (DECISIONS.md, answers to Q10-Q12): 1-6 accepted as implemented.

### Q7 (2026-11-18): training and evaluation settings the code needs before real runs (Point 9)

- Affects: `trustmap/train.py` (`TRAIN_RHO_RANGE`, `VALIDATION_EPISODES`, `VALIDATION_SEED`), `trustmap/ppo.py` `PPOPolicy` (`deterministic`), `trustmap/evaluate.py` and `trustmap/evaluation.py` `make_episodes`.
- Question: Points 7 and 8 are locked, but running them needs five settings DECISIONS.md does not give yet. Each is implemented as a clearly marked placeholder and written into every run's `config.json`.
  1. Training mix of ρ. The proposal says "trained with ρ uniform in [0, 0.8]". Placeholder: ρ drawn uniformly from [0, 0.8] at every reset, so one run sees all corruption levels.
  2. Validation set (7d checkpoint selection). Placeholder: 1,000 fixed episodes from the validation houses, ρ drawn per episode from the training range, same κ, λ and placement as the run, seed 100,000.
  3. Greedy or sampled actions at evaluation. Placeholder: greedy (the highest-probability room). Sampling would add noise to every metric; greedy is the usual choice for reporting a trained policy.
  4. Test episodes. Placeholder: for each ρ, 200 episodes, house drawn uniformly from the 200 test houses, seed 200,000 for every method and every ρ (paired evaluation).
  5. When validation runs inside a rollout. SB3 collects 16,384 decisions per update (2,048 × 8 copies), so the "every 50,000 decisions" check lands partway through a rollout and evaluates the policy from the last update. A 1M run actually ends at 1,015,808 decisions (62 whole rollouts).
  6. Bayes-nominal edge case (`trustmap/baselines.py`). If the displayed c rounds to exactly 1 while the claim is wrong, the nominal belief leaves no probability for any other room. Placeholder: fall back to the nearest unsearched room. This needs r within about 10⁻⁶ of 1 with κ > 1, so it should essentially never happen.
- Recommendation: accept 1–6 as written; they follow the proposal and the 7d text.
- Status: resolved 2026-11-18 (DECISIONS.md, answers to Q7-Q9, 9a, 9b): all as implemented, except the test set, now 5,000 episodes per rho, exactly 25 per test house (`evaluation.make_test_episodes`).

### Q8 (2026-11-18): "success within 1 decision" penalizes optimal on-the-way searches (Point 6b/9, observation)

- Affects: how E2 reads `success_first_decision` in `trustmap/metrics.py`; no code change needed.
- Observation: on 400 stand-in test episodes at ρ = 0.2 (mean r 0.8), the exact optimum's first search was not the claimed room in 147 of them (37%). In those cases the room searched first lies almost exactly on the way to the claimed room (median detour 2%) and holds the object with a probability around 6%: searching it first costs almost nothing, so the optimum does it, then continues to the claim. As a result, the optimum's first-decision success (0.56) was lower than the blind greedy follower's (0.82). Rechecked on the generated test houses (A* distances): at ρ = 0.2 the optimum skips the claimed room first in 99 of 400 episodes (25%), at ρ = 0.4 in 178 of 400 (45%), with a median detour of about 10%, so the effect is real in the actual houses too, though smaller than in the stand-ins. 6b calls "success within 1 decision" the clearest sign of correct trust, but by itself it would rate the optimal policy as trusting the map too little.
- Options: (a) keep it as a secondary metric and always show it next to the optimum's own value; (b) replace it in E2 with "agreement with the optimum's first room" (already in the proposal's E2) or "the claimed room was searched before any room off the way"; (c) drop it.
- Recommendation: (a) together with (b): report it only alongside the optimum's value, and use agreement with the optimum's first choice as the trust measure in E2.
- Status: resolved 2026-11-18: first-decision success stays a secondary metric shown next to the optimum's own value; the trust measure is the 9c follow rate against c - c* (E2).

### Q9 (2026-11-18): small generator choices not spelled out in the engineering parameters

- Affects: `trustmap/generator.py` (`_draw_geometry`, `_place_doors`, `_neighbours`, `generate_house`, `object_candidates`).
- Question: implementing the locked generator needed these choices. All of them follow Khaled's "simplest option that works" rule, and none touches the trust question. Please confirm or change them.
  1. The public zone always takes the low-coordinate side of the first cut. A mirrored house has identical distances, so this changes nothing.
  2. The order of rooms inside each zone is shuffled per house (otherwise, for example, the kitchen would always sit at the same end of the public zone).
  3. A private room that touches several public rooms gets its one door to a random one of them; the same for a private room choosing among several connected private rooms.
  4. Two rooms count as neighbours (a door is possible) only when their shared wall is at least 1.2 m long, so the centred 1.0 m door stays clear of the 0.1 m half-walls at each end.
  5. On rejection only the geometry is redrawn, not the room types, so the fifth-room shares stay exactly 50/25/25 (redrawing the types too would make the rarely-valid specs rarer). The training pool came out bedroom 2,508, office 1,276, dining 1,216.
  6. Room borders, cuts and door centres are rounded to the 0.10 m grid, so every wall is exactly two cells thick and every door exactly ten cells wide.
  7. The 2D grid has no furniture yet. The engineering parameters do not mention furniture in the grid; the Gazebo world has simple box furniture (Point 2). If furniture goes into the Gazebo world, it should also go into the shared grid so that A* and the camera agree in both simulators.
  8. Object placement also requires the object to be at least the robot radius (0.2 m) from the viewpoint (not under the robot) and not visible from any other room's viewpoint. The second part is what makes "impossible elsewhere" true; without it, an object near a door could be seen from a small neighbouring room.
- Recommendation: accept 1–6 and 8 as written; for 7, keep the 2D grid without furniture and put no furniture where it would block a viewpoint or a path.
- Status: resolved 2026-11-18: 1-6 and 8 accepted; 7: no furniture in either simulator, rooms told apart by floor colour, target as a visual-only model in Gazebo.

### Q2 (2026-11-17): house generator parameters (Point 3a)

- Affects: the house generator (not yet written), so nothing generated by it exists yet.
- Question: 3a fixes the method (ProcTHOR-style room-spec tree, public zone kitchen + living (+ dining), private zone bedroom(s) + bathroom (+ office), growth weights, 1.0 m doors, bathroom one door, open connection only kitchen–living, entrance in kitchen or living, rejection sampling). These numbers and rules are still needed:
  1. Footprint size range (width and depth in meters), probability of one or two corner cuts, and cut size.
  2. Growth weight of each room type, the split order (zones first, then rooms), how the split direction is chosen, and a minimum room width.
  3. Door graph: which room pairs must or may be connected by doors, apart from the bathroom and kitchen–living rules (for example, must every private room open onto the living room?). Which adjacency rules make a house rejected.
  4. Where a door sits on a shared wall (centered or random).
  5. What the entrance is used for, since the start room is drawn by the 3c recipe, not at the entrance.
- Recommendation: decide these in one message in the RL chat, using ProcTHOR's defaults where they exist. I can draft a full parameter proposal if that helps.
- Status: resolved 2026-11-18 (DECISIONS.md, engineering parameters). Implemented in `trustmap/generator.py`; the small choices it still needed are in Q9.

### Q3 (2026-11-17): occupancy grid, A* and viewpoints

- Affects: the occupancy grid and planned-distance table inside the generator, and the shared A* used by both simulators.
- Question: grid cell size and wall thickness; whether A* moves in 4 or 8 directions (8 directions with diagonal cost √2 gives shorter, more realistic lengths); whether walls are inflated by the robot radius before planning, and by how much; where each room's viewpoint is (for example the free cell nearest the room center).
- Recommendation: 0.05 m cells, 8-connected A*, inflate by the robot radius used in Gazebo, viewpoint at the free cell nearest the room center.
- Update 2026-11-18: `trustmap/grid.py` and `trustmap/planning.py` now exist with cell size, connectivity and inflation radius as required arguments (no defaults), so answering Q3 only fills in numbers. Two small rules were implemented provisionally and can be flipped if the RL chat disagrees: (i) 8-connected moves may not cut a wall corner (a diagonal step needs both side cells free); (ii) inflation blocks every cell whose center is within the radius of a blocked cell's center. Speed, pure Python: one 5×5 distance table for a 12 m × 10 m house takes about 1.8 s at 0.05 m cells and 0.4 s at 0.10 m cells, so precomputing the ~5,400 pool houses once takes about 2.8 h or 40 min. Either is acceptable since it runs once and is saved; 0.10 m is worth considering if Nav2's map resolution allows it.
- Status: resolved 2026-11-18 (0.10 m cells, 0.20 m walls, 8-connected A*, 0.20 m inflation, viewpoint at the free cell nearest the room centre). Implemented in `trustmap/generator.py` with `grid.py` and `planning.py`. Pool build measured: 5,000 training houses in 9.3 min on 8 cores.

### Q4 (2026-11-17): virtual camera and object position inside a room

- Affects: the shared virtual camera (field-of-view cone plus line-of-sight ray casting).
- Question: the camera's field-of-view angle; the robot's heading on arrival at a viewpoint (fixed, facing the room center, or a 360° turn); where the target object sits inside its room. This matters because the exact optimum assumes that searching the object's room always finds it ("detection is deterministic"), so the object must always be visible from the viewpoint within about 1.5 m.
- Options: (a) a 360° look at the viewpoint and the object placed within 1.5 m with clear line of sight; (b) a fixed field of view and the object placed inside the view cone.
- Recommendation: (a), because it guarantees detection in the object's room and keeps the optimum exact.
- Update 2026-11-18: `trustmap/camera.py` now exists: view cone, range (1.5 m from the Agreements) and line-of-sight ray cast; the field-of-view angle is a required argument (2π = 360° look). Two rules were implemented provisionally: (i) the target's own cell is not checked for walls, so an object standing on furniture (a blocked cell) can still be seen if nothing else is in the way; (ii) a ray passing exactly through a cell corner checks both cells beside the corner. Still needed: the angle, the heading on arrival, and the object placement rule.
- Status: resolved 2026-11-18 (detection within 1.5 m with line of sight, 360° look, object on a random valid cell of its room). Implemented in `trustmap/generator.py` `object_candidates` / `place_object`.

### Q5 (2026-11-17): the single development house

- Affects: development setup before the pool is used (3a: "development starts in one fixed house").
- Question: should it be generator seed 0, or a hand-made house?
- Recommendation: generator seed 0, so it uses the same code path as the pool.
- Status: resolved 2026-11-18 (generator seed 0, checked by eye with `python -m trustmap.plot_house 0`).

### Q6 (2026-11-17): exact definition of decision regret

- Affects: `trustmap/metrics.py` (regret not yet written), using `trustmap/optimum.py`.
- Question: 6b lists decision regret as a metric but does not define it. Two choices are needed. (a) Expected or realized: regret = V*(start) − V^π(start), both computed exactly from the episode's belief (the policy's expected return is found by evaluating its choices over the same 160 states), or regret = V*(start) − the return the policy actually got in that episode (noisy, since luck decides where the object was). (b) Which probability the reference optimum uses for "claim correct": the true r (the optimum of the real generator; with κ known per condition, c maps one-to-one to r, so this is the best any policy seeing c can do) or the displayed c.
- Recommendation: (a) expected regret, computed exactly, reported in meters by multiplying by the house's largest distance; (b) the true r. `solve_episode` already defaults to r and accepts any other value.
- Status: resolved 2026-11-17 in DECISIONS.md 8a/8b (expected decision regret per decision, summed, in meters; optimum with the true r). Implemented 2026-11-18 in `trustmap/metrics.py` `decision_regret`.

### Q1 (2026-11-17): scope of `placement = uniform`

- Affects: `trustmap/cooccurrence.py` `room_weights`, used by `trustmap/episode.py` `generate_episode`.
- Question: 3b says "uniform is just the table with every weight set to 1", but the table is used in three places: the wrong-claim object draw, the table branch of the λ claim draw, and the habit feature (4a). Which of these become all-ones?
- Options: all three; placement only; placement plus habit.
- Recommendation: all three, so the ablation world has no commonsense at all.
- Status: resolved 2026-11-17 (all three; recorded in DECISIONS.md 3b).

