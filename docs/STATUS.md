# STATUS

Written by Claude Code only. Updated at the end of every coding session.

## Results in two minutes (2026-11-19)

All locked runs are trained (21 policies in `runs/`), every experiment has been run on the 5,000-episode test set (9b), and the Gazebo evaluation (9e) is done. Regret is in meters above the exact value-iteration optimum, as the mean ± 95% t-interval over training seeds. Figures are in `results/figures/` (PNG and PDF).

**H1 (PPO-Conf beats ignoring the map and blindly following it): supported.** E1 regret, `e1_regret`:

| ρ | Random | Nearest | Follower | NoPrior | Binary | Conf |
|---|---|---|---|---|---|---|
| 0 | 13.68 | 7.59 | 0.00 | 7.15 ± 0.06 | 0.93 ± 0.09 | 0.00 |
| 0.2 | 11.90 | 5.76 | 0.71 | 5.11 ± 0.04 | 0.50 ± 0.02 | 0.20 ± 0.02 |
| 0.4 | 9.86 | 3.84 | 2.01 | 3.00 ± 0.03 | 0.56 ± 0.04 | 0.43 ± 0.04 |
| 0.6 | 8.90 | 2.61 | 3.84 | 1.82 ± 0.10 | 1.22 ± 0.09 | 0.51 ± 0.06 |
| 0.8 | 8.81 | 2.50 | 6.57 | 1.65 ± 0.05 | 3.00 ± 0.18 | 0.65 ± 0.07 |

Bayes-nominal is 0 at every ρ (it is the optimum when κ = 1). Conf is below NoPrior at every ρ and below the follower at every ρ > 0 (both are 0 at ρ = 0). The follower crosses NoPrior between ρ = 0.4 and 0.6, and Binary crosses it between 0.6 and 0.8.

**H2 (trust switches at c*): partly supported.** E2 (`e2_trust`; 25,000 episodes, 1,911 without a c*, median c* 0.46): PPO-Conf follows the claim at a rate of 0.43 when c − c* = −0.075, 0.55 at −0.025, 0.71 at +0.025 and 0.84 at +0.075, reaching about 0.98 by +0.175. The switch is centred on c* but is a smooth ramp about ±0.15 wide rather than the optimum's step. Some distance dependence remains: at −0.025, far claims are followed at 0.67 against 0.42 for near ones. Binary's follow rate is flat against c (about 0.67) but rises with c − c*, because c* itself depends on the house and the target.

**H3 (calibration mismatch costs regret): mostly supported.** E3 2×2, `e3_calibration` (regret averaged over ρ = 0.2–0.8):

| | test κ = 1 | test κ = 3 |
|---|---|---|
| train κ = 1 | 0.45 | 0.78 |
| train κ = 3 | 0.61 | 0.53 |
| Bayes-nominal | 0 | 0.67 |

Both mismatched cells are worse than the matching cells. The caveats: train κ = 3 tested at κ = 1 is only worse at ρ ≥ 0.6 (at ρ = 0.8 it is 1.14 against 0.65), and at test κ = 3 Bayes-nominal beats both PPO policies for ρ ≤ 0.4 (0.13 and 0.31 against 0.22 and 0.47 for train κ = 3); at ρ ≥ 0.6 the train κ = 3 policy is better (0.54 and 0.88 against 0.96 and 1.27).

**H4 (the map helps more for weak-habit targets): mostly not supported.** The NoPrior − Conf gap (`h4_habit_gap`), weak against strong habit: ρ 0: 7.45 ± 0.07 vs 6.83 ± 0.07; 0.2: 4.97 ± 0.06 vs 4.85 ± 0.10; 0.4: 2.40 ± 0.05 vs 2.74 ± 0.07; 0.6: 1.08 ± 0.10 vs 1.54 ± 0.08; 0.8: 0.72 ± 0.09 vs 1.29 ± 0.10. The predicted direction holds only at ρ ≤ 0.2 and is reversed at ρ ≥ 0.4. A possible reason: for a strong-habit target, a wrong claim usually points away from the habit room, and seeing the claim's low confidence still tells the policy which room is probably not the answer.

**λ row (evaluation only, Q10a).** The conclusions hold at λ = 0 and λ = 1. PPO-Conf: 0.00, 0.21, 0.45, 0.52, 0.66 at λ = 0 and 0.00, 0.20, 0.40, 0.50, 0.61 at λ = 1 (ρ = 0 … 0.8). NoPrior: 5.56, 3.74, 2.24, 1.51, 1.83 and 8.70, 6.31, 3.71, 2.12, 1.45. In both, the follower crosses NoPrior between ρ = 0.4 and 0.6 and Binary between 0.6 and 0.8.

**Uniform placement row (PPO-Conf only, Q10b).** 0.01, 0.19 ± 0.01, 0.36 ± 0.03, 0.34 ± 0.01, 0.53 ± 0.06. The follower crosses Nearest between ρ = 0.4 and 0.6.

**Gazebo (9e; houses 200,000 and 200,001, ρ = 0.4, 40 episodes per policy, `results/gazebo/summary.csv`).**

| | success | SPL | regret (m) | planned (m) | same as 2D | driven (m) | overhead | nav failure / goal | end-to-end success | camera agrees |
|---|---|---|---|---|---|---|---|---|---|---|
| PPO-Conf | 0.95 | 0.745 | 0.40 | 11.90 | 1.0 | 11.94 | +0.4% | 0.013 (1 episode) | 0.90 | 0.975 |
| Follower | 0.70 | 0.576 | 2.91 | 12.58 | 1.0 | 12.76 | +1.4% | 0.013 (1 episode) | 0.65 | 0.975 |

Every Gazebo decision equals the 2D decision. The two end-to-end losses per policy are 200,000 episode 15 (Nav2 failed the same goal twice and the robot was teleported) and 200,001 episode 1 (the object is 1.487 m from the viewpoint, at the edge of the 1.5 m camera range, and the robot stopped at a pose from which it cannot be seen).

**Demo set (DECISIONS "Demo material").** 10 hand-picked PPO-Conf episodes in Gazebo, chosen by a stated rule (`results/demo/README.md`): 8 successes and 2 failures at ρ = 0.4, 8 different targets (laptop, alarm clock, remote control, book, mug, towel, cell phone, keys), houses 200,000–200,003 (2 or 4 per house). Each has an MP4 with the overhead view and the synchronised side panel, an `episode.json`, and for every success a still from the robot camera after turning to the object. Plus `results/demo/demo_2d.gif` and `demo_2d_follower.gif` (2D backup). They are not part of the result rows.

**Surprising or broken.** (1) H4 is reversed at ρ ≥ 0.4. (2) Bayes-nominal beats PPO under κ = 3 at low ρ. (3) The camera range-edge miss in 200,001 episode 1 is a real end-to-end effect: the decision rule counts the room as searched, but the camera at the actual stopping pose cannot see the object (33 of 72 poses within 0.15 m of the viewpoint cannot see it). In the demo, 200,001 episode 1 is therefore a decision success whose camera did not report the object. (4) Two concurrent video builds collided once and truncated the 200,000 episode 1 video; that episode was re-recorded and rebuilt. Nothing else is broken.

## Code state

All eleven points are locked and implemented in `trustmap/` (house generator, episodes, observations, search rules, optimum and c*, metrics, baselines, Q-learning check, env and PPO training, E0–E3, Gazebo export/runner/summary, policy export, demo selection/recording/video, 2D animation). 63 pytest tests pass (`.venv`, Python 3.14). Committed: all CSVs and figures in `results/`, each run's best/final model, config and validation curve, the house pools, the Gazebo logs and the demo material. Q10–Q12 are resolved; OPEN_QUESTIONS has no open items.

Not done: the ProcTHOR-10K row (optional), E4 (stretch), the report text and the slides.

## Next coding step

None is required for the locked plan. Next is writing: fill `report/` from the numbers above, check `report/refs.bib`, and make the slides (the demo MP4s and stills are ready for them). Optional: ProcTHOR-10K row, E4.

## Log

- 2026-11-17: repository set up; Points 3b, 3c, 4, 5, 6, exact optimum and metrics implemented.
- 2026-11-18: shared geometry; Points 7–8 (env, training, regret, baselines, E0 test); house generator and pools; Q7–Q9 added.
- 2026-11-19: Points 9–11: c*, E0–E3, all locked runs trained, Gazebo tested, README/report skeleton; Q10–Q12 added, then resolved; λ row and uniform row; Gazebo evaluation 9e; demo material; results summary.
