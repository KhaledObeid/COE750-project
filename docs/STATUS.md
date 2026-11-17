# STATUS

Written by Claude Code only. Updated at the end of every coding session.

## Current state (2026-11-17)

Built, in the `trustmap` package:

- `cooccurrence.py` (Point 3b): the six room types, the eight targets with their habit strength, the co-occurrence table, and `room_weights`. With `placement = uniform` the whole table becomes ones (OPEN_QUESTIONS Q1, answered by Khaled).
- `house.py` (Points 3a/3b): a `House` holding the 5 room types and the 5×5 planned-distance table; the fifth-room draw (second bedroom 50%, office 25%, dining 25%); the seed ranges of the train, validation and test pools.
- `episode.py` (Point 3c): `EpisodeConfig` (ρ, κ, λ = 0.5, placement, target set) and `generate_episode`, which runs recipe steps 2–6 in the locked order and also draws the per-episode row shuffle (4a).
- `observation.py` (Point 4): the 5×5 observation (distance scaled by the house's largest distance, searched, claimed, c, habit) in shuffled row order, plus the PPO-Conf / PPO-Binary / PPO-NoPrior input masks.

Two input checks added, which do not change behavior inside the decided ranges: ρ must be in [0, 1), because ρ = 1 gives an invalid Beta, and κ must be positive.

Tested with `pytest` (17 tests pass) against the DECISIONS.md worked numbers: the towel habit (8/13.5), the claim distribution at λ = 0.5 (0.40 / 0.17 / 0.17 / 0.14 / 0.12), the Beta(6, 4) spread, the start room never being the claimed room and being uniform over the other four, P(claim correct) = r, the wrong-claim placement (8/12 bathroom), uniform mode, and the variant masks.

Not built, because it is not decided: the house generator geometry, the occupancy grid, A*, viewpoints and the virtual camera (OPEN_QUESTIONS Q2–Q5). Step 1 of the recipe (draw a house from the pool), action handling and the "searched" update (Point 5), reward (Point 6), value iteration, the Gymnasium environment, baselines and PPO are also not built. The tests use a made-up 5×5 distance table, not a generated house.

## Next coding step

The house generator (Point 3a), once Q2–Q3 are answered in the RL chat.

## Log

- 2026-11-17: repository set up from the claude.ai RL Project chat.
- 2026-11-17: Points 3b, 3c and 4 implemented and tested; generator questions logged.
