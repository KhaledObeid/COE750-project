# CLAUDE.md — COE750 project "Should the Robot Trust Its Map?"

Read this whole file at the start of every session, then read `docs/DECISIONS.md`, `docs/STATUS.md` and `docs/OPEN_QUESTIONS.md` before doing any task.

## What this project is

A solo reinforcement learning course project by Khaled Obeid (COE750, the master's section of COE550, Dr. Fouad Trad, LAU). A robot searches a 5-room house for a target object using a semantic prior map that claims one room with a confidence c; the map is wrong a controlled fraction of the time (corruption ρ) and its confidence may be miscalibrated (κ). The research question is whether a PPO policy learns the right amount of trust in the map, measured against an exact value-iteration optimum. Training runs in our own fast 2D floor-plan simulator; the frozen policy is tested in Gazebo Fortress with Nav2. Target finish date: 30 November 2026. Deliverables: IEEE-format report, slides, and this code with a README (code is 15% of the grade).

## How the three of us work together

Decisions are made in a separate claude.ai chat (the "RL" Project), not here. That chat is the only writer of `docs/DECISIONS.md`. Claude Code (you) implements, and is the only writer of `docs/STATUS.md` and `docs/OPEN_QUESTIONS.md`. Khaled moves information between the two by pulling and pushing this repository. Each file has one writer so nothing is overwritten.

Rules for Claude Code:

1. Every decision in `docs/DECISIONS.md` under "Locked decisions" is fixed. Implement it exactly. Never change, reinterpret, or "improve" a locked decision in code, even if another choice looks better; if you think a decision is wrong or unworkable, explain why to Khaled and add it to `docs/OPEN_QUESTIONS.md`.
2. Items under "Proposed but not yet locked" and every point not yet reached (see "Next step" in `DECISIONS.md`) are not decided. Do not implement them beyond clearly marked placeholders or config stubs, and say so.
3. When the code forces a choice that `DECISIONS.md` does not cover (a number, an interface, an edge case), stop and ask Khaled before choosing. Then append the question to `docs/OPEN_QUESTIONS.md` with the date, the file and function it affects, the options you see, and your recommendation. Small purely technical choices that do not change behavior or results (variable names, file layout, helper functions) you may make yourself.
4. Never edit `docs/DECISIONS.md`.
5. At the end of every session, update `docs/STATUS.md`: what was built, what was tested and how, what is broken or incomplete, and the next coding step. Keep it short and current; replace stale lines rather than piling up history (a dated one-line log at the bottom is fine).
6. Commit in small steps with clear messages that name the decision point they implement, for example "env: episode generation recipe (Point 3c)".

## Khaled's background and how to explain

Khaled knows Python, ROS2, C++, OpenCV and embedded systems well, and from the RL course he knows only: Gymnasium basics, MDPs, state and action values, policy and value iteration, Monte Carlo, TD learning (SARSA, Q-learning, Expected SARSA), and bandits with ε-greedy. Anything beyond that (PPO, neural network policies, action masking, advantage estimation, vectorized environments, and so on) must be explained in plain words the first time it appears, both in your reply and in a short code comment. He must be able to explain every part of the code in his presentation, so prefer clear, readable code over clever code. Write replies in direct full-sentence paragraphs, without introductions or emojis.

## Technical setup

Linux machine: Ubuntu 22.04, ROS2 Humble, Gazebo Sim 6.18.0 (Fortress). Fortress uses the "ignition" names (`ign gazebo`, plugins such as ignition-gazebo-diff-drive-system, `ignition.msgs.*` in ros_gz bridge configs); follow Fortress/Humble tutorials, not Harmonic ones. Bridge from apt: ros-humble-ros-gz. Khaled also works on a Windows machine; the 2D simulator and training code must run on both (pure Python, CPU).

## Engineering requirements that come from the decisions

The 2D simulator and Gazebo must share the same code for observations, distances (A* on the same occupancy grid) and the virtual camera (field-of-view cone plus line-of-sight ray casting). All randomness happens at environment reset and is seeded; house pools are fixed by seed range (training 0–4,999, validation from 100,000, test from 200,000). Every dial is a config value with an off switch: ρ, κ, λ, `placement = table | uniform`, the target set, and the variant mask (PPO-Conf, PPO-Binary, PPO-NoPrior). Gazebo results never feed back into training.

## Suggested build order (only for parts already locked)

1. House generator (Point 3a) with occupancy grid, room viewpoints, and the precomputed planned-distance table; a quick plot to check houses by eye.
2. Episode generation recipe (Point 3c, final recipe section) with the co-occurrence table (Point 3b).
3. Observation builder (Point 4): 5 shuffled rows of 5 features, plus the variant masks.
4. Exact optimum by value iteration over current room × searched set (needed for regret), once Points 5 and 6 are locked.
5. The Gymnasium environment, baselines and PPO training only after Points 5–7 are locked.
