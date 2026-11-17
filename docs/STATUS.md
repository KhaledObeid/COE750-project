# STATUS

Written by Claude Code only. Updated at the end of every coding session.

## Current state (2026-11-17)

Built, in the `trustmap` package:

- `cooccurrence.py` (Point 3b): the six room types, the eight targets with their habit strength, the co-occurrence table, and `room_weights`. With `placement = uniform` the whole table becomes ones (3b, scope of uniform).
- `house.py` (Points 3a/3b): a `House` holding the 5 room types and the 5×5 planned-distance table; the fifth-room draw (second bedroom 50%, office 25%, dining 25%); the pool seed ranges.
- `episode.py` (Points 1, 3c, 5a, 6b): `EpisodeConfig` (ρ, κ, λ = 0.5, placement, target set, decision cap = 3) and `generate_episode`, which runs recipe steps 2–6 in the locked order, draws the per-episode row shuffle (4a), and checks at runtime that the object is never in the start room (5a).
- `observation.py` (Point 4): the 5×5 observation in shuffled row order, plus the PPO-Conf / PPO-Binary / PPO-NoPrior input masks.
- `search.py` (Points 5, 6): `RoomSearch`, the room-level rules shared by the future Gymnasium env and the Gazebo runner. It provides the action mask over rows (unsearched rooms only), automatic success when the chosen room holds the target, reward −distance / house max distance, the 3-decision cap with a −1 failure penalty, and γ = 1 (plain sum).
- `optimum.py` (exact optimum): value iteration over current room × searched set (160 states) with the Point 6 reward and the cap. The reset belief puts r on the claimed room and spreads 1 − r over the three other non-start rooms by habit. `solve_episode` uses the true r by default (Q6).
- `metrics.py` (6b): success, SPL (failures count 0; shortest = planned distance from the start room to the object's room), success on the first decision, decisions, meters.

Input checks that do not change behavior inside the decided ranges: ρ in [0, 1), κ > 0, cap between 1 and 4.

Tested with `pytest` (27 tests pass on Python 3.14 and 3.10). The tests cover the DECISIONS.md worked numbers for 3b/3c, the observation layout and masks, masking/success/failure/reward rules, the metrics, and a brute-force check that V* equals the best of every possible room order played through `RoomSearch` for caps 2, 3 and 4. All tests use a made-up distance table, not a generated house.

Not built because it is not decided: the house generator, occupancy grid, A*, viewpoints and virtual camera (Q2–Q5); decision regret (Q6). Not built because Point 7 is not locked: the Gymnasium environment, the policy network and PPO training. Baselines are still only "proposed", so they are not built either.

## Next coding step

The house generator (Point 3a), once Q2–Q3 are answered in the RL chat; regret once Q6 is answered. After Point 7, wrap `RoomSearch` and the observation builder in a Gymnasium environment.

## Log

- 2026-11-17: repository set up from the claude.ai RL Project chat.
- 2026-11-17: Points 3b, 3c and 4 implemented and tested; generator questions logged.
- 2026-11-17: Points 5 and 6 (masking, episode end, reward, cap), exact optimum and 6b metrics implemented and tested; Q1 resolved, Q6 (regret definition) added.
