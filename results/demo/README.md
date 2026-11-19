# Demo episodes (Gazebo, PPO-Conf)

**These are hand-picked illustrative examples. They are not part of the Gazebo result rows** (those come only from `trustmap.gazebo_runner`, see `results/gazebo/summary.csv`). They follow the DECISIONS paragraph "Demo material (Khaled, 2026-11-19; refines 9e and 11b)".

## How the episodes were picked

The candidates are the 25 E1 test episodes (9b, seed 200,000) of each demo house, 200,000, 200,001, 200,002 and 200,003, at ρ = 0.4 and κ = 1. That makes 100 candidates. Each one was played in 2D by the exported PPO-Conf policy (`gazebo/policies/ppo_conf.npz`, the median-validation seed); Gazebo decisions are identical to 2D, so the 2D outcome is the Gazebo decision outcome. The rule (`python -m trustmap.demo_select`, written to `selection.json`) is:

- successes: go through the houses in order; in each house take the first 2 successful episodes (by episode index) whose target has not already been selected, falling back to any target if none is left;
- failures: take the first 2 failed episodes in scan order (house order, then episode index) at ρ = 0.4, falling back to ρ = 0.6 and then 0.8 if there are fewer than 2.

8 of the 100 candidates fail at ρ = 0.4, so no fallback was needed. The result is 10 episodes: 8 successes with 8 different targets and 2 failures, covering all 4 houses.

## Episodes

| house | episode | target | ρ | claim c (claim) | object room | rooms searched | outcome | planned / driven (m) | regret (m) | camera saw object | files |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 200000 | 0 | laptop | 0.4 | 0.69 (correct) | kitchen | kitchen | success | 2.4 / 2.3 | 0.00 | yes | [video](200000_0/video.mp4), [still](200000_0/still.png), [json](200000_0/episode.json) |
| 200000 | 1 | alarm clock | 0.4 | 0.33 (wrong) | bedroom | bathroom → bedroom | success | 19.9 / 20.3 | 0.93 | yes | [video](200000_1/video.mp4), [still](200000_1/still.png), [json](200000_1/episode.json) |
| 200000 | 11 | book | 0.4 | 0.64 (wrong) | bathroom | living → kitchen → dining | **failure** | 11.4 / 11.5 | 0.00 | no | [video](200000_11/video.mp4), [json](200000_11/episode.json) |
| 200000 | 20 | alarm clock | 0.4 | 0.53 (wrong) | living | dining → kitchen → bedroom | **failure** | 15.6 / 15.5 | 1.78 | no | [video](200000_20/video.mp4), [json](200000_20/episode.json) |
| 200001 | 1 | remote control | 0.4 | 0.75 (correct) | bedroom 2 | living → bedroom 2 | success | 15.3 / 15.3 | 0.18 | no | [video](200001_1/video.mp4), [still](200001_1/still.png), [json](200001_1/episode.json) |
| 200001 | 2 | book | 0.4 | 0.74 (correct) | bathroom | bathroom | success | 8.5 / 9.5 | 0.00 | yes | [video](200001_2/video.mp4), [still](200001_2/still.png), [json](200001_2/episode.json) |
| 200002 | 1 | mug | 0.4 | 0.38 (correct) | bedroom | kitchen → living → bedroom | success | 19.5 / 19.9 | 0.00 | yes | [video](200002_1/video.mp4), [still](200002_1/still.png), [json](200002_1/episode.json) |
| 200002 | 2 | towel | 0.4 | 0.55 (wrong) | kitchen | living → bathroom → kitchen | success | 22.6 / 22.4 | 2.80 | yes | [video](200002_2/video.mp4), [still](200002_2/still.png), [json](200002_2/episode.json) |
| 200003 | 1 | cell phone | 0.4 | 0.63 (correct) | kitchen | kitchen | success | 3.7 / 3.7 | 0.00 | yes | [video](200003_1/video.mp4), [still](200003_1/still.png), [json](200003_1/episode.json) |
| 200003 | 12 | keys | 0.4 | 0.42 (correct) | dining | dining | success | 3.7 / 3.7 | 0.15 | yes | [video](200003_12/video.mp4), [still](200003_12/still.png), [json](200003_12/episode.json) |

"Camera saw object" is the virtual camera (field-of-view cone plus line of sight) at the robot's actual pose after its 360-degree look. In 200,001 episode 1 the decision outcome is "found", but the object sits 1.49 m from the viewpoint, at the edge of the 1.5 m camera range, and the robot stopped where the camera cannot see it. The Gazebo evaluation counts the same episode as an end-to-end miss. The still for that episode is still taken after turning toward the object. Episode 200,000/11 is a failure with zero regret: the claim (c = 0.64) was wrong, and the optimum would have searched the same three rooms in the same order, so this miss was unavoidable in expectation.

## What each file shows

`video.mp4` (1280 × 720, H.264, 3× simulation speed) shows the fixed overhead camera on the left. The room labels are drawn on top, the claimed room is marked "map: here, c = …", and a red ring marks the object. On the right is a panel synchronised to the simulation timestamps:

- what the robot is told at the start: target, start room, claimed room and confidence c, ρ, κ, and each room's habit weight for the target;
- a box "hidden from the robot": the true reliability r, the object's real room, and whether the claim is correct;
- one line per decision as it happens: the chosen room, the optimum's choice in the same state, that choice's regret in meters, then the outcome (FOUND, not here, or FAILURE after the third search);
- at the end, planned and driven distance and the episode's regret.

`still.png` (successes only) is one picture from the robot's camera (`/camera/image`) after the robot detected the object and turned toward it. `episode.json` holds everything the panel uses. `logs/` holds the Gazebo launch and runner logs. The 200,000 episode 1 video was truncated by an accidental second build running at the same time, so that episode was re-recorded on its own (`logs/rerun_200000_1/`); the episode is the same because placement and policy are seeded.

`demo_2d.gif` and `demo_2d_follower.gif` are the 2D backup animation of test house 200,000 episode 8 at ρ = 0.4. PPO-Conf finds the object (7.6 m), while the greedy map follower fails (18.1 m).

## Regenerate

See the "Demo videos" section of the repository README: `python -m trustmap.demo_select`, then `gazebo/run_house.sh <house> results/demo/logs --module trustmap.gazebo_demo -- --selection results/demo/selection.json` for each house, then `python -m trustmap.demo_video`. The overhead frames are deleted after the MP4 is built and are not committed.
