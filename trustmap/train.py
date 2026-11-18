"""Train one PPO agent (Points 7a-7d). Run:  python -m trustmap.train --variant conf --seed 0

What one run does:
  1. builds 8 copies of the environment (vectorized environments: SB3 steps
     all 8 in turn and collects their decisions into one batch, so each
     update sees more varied data),
  2. trains MaskablePPO with the 7c settings for 1 million decisions (7d),
  3. every 50,000 decisions pauses, plays a fixed set of validation-house
     episodes, and records the mean decision regret in meters (8b); the
     checkpoint with the lowest validation regret is saved as best_model.zip
     and is the one tested later (7d). The recorded values form the learning
     curve (validation_curve.csv).

A seed fixes all randomness of one run: starting network weights, the
episodes drawn by the 8 environment copies, and PPO's own sampling.

SB3 trains in whole rollouts of n_steps x 8 = 16,384 decisions, so a 1M run
actually stops at 1,015,808 decisions (62 rollouts).
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from trustmap.episode import DEFAULT_LAMBDA, DEFAULT_MAX_DECISIONS, EpisodeConfig
from trustmap.env import TrustMapEnv
from trustmap.evaluation import evaluate, make_episodes, solve_all, summarize
from trustmap.observation import Variant
from trustmap.ppo import NUM_ENVS, POLICY_KWARGS, PPO_SETTINGS, PPOPolicy
from trustmap.pool import load_pool
from trustmap.smoke_houses import smoke_pool

TOTAL_DECISIONS = 1_000_000  # 7d
EVAL_EVERY = 50_000          # 7d

# Point 9a: one policy per variant and seed, trained with rho drawn uniformly
# from [0, 0.8] fresh every episode.
TRAIN_RHO_RANGE = (0.0, 0.8)
# Q7 (answered 2026-11-18): 1,000 fixed validation episodes from the validation
# houses, rho drawn from the training range, the run's kappa, lambda and
# placement, seed 100,000. Greedy actions (most probable room) at evaluation.
VALIDATION_EPISODES = 1000
VALIDATION_SEED = 100_000

VARIANTS = {"conf": Variant.CONF, "binary": Variant.BINARY, "noprior": Variant.NOPRIOR}


def run_name(variant_key, seed, kappa=1.0, lam=DEFAULT_LAMBDA, placement="table"):
    """Folder name of one run under runs/, e.g. conf_kappa1_lam0.5_table_seed0."""
    return f"{variant_key}_kappa{kappa:g}_lam{lam:g}_{placement}_seed{seed}"


def load_houses(source, pool):
    """'pool' = the generated, saved house pool (3a); 'smoke' = stand-in houses for debugging only."""
    if source == "smoke":
        return smoke_pool(pool)
    return load_pool(pool)


class ValidationCallback(BaseCallback):
    """Every `every` decisions: validation regret, learning-curve row, best checkpoint."""

    def __init__(self, episodes, max_decisions, variant, every, out_dir):
        super().__init__()
        self.episodes = episodes
        self.solutions = solve_all(episodes, max_decisions)  # solved once, reused at every check
        self.max_decisions = max_decisions
        self.variant = variant
        self.every = every
        self.out_dir = out_dir
        self.next_check = 0
        self.best_regret = float("inf")
        self.curve_path = out_dir / "validation_curve.csv"
        self.start_time = time.time()

    def _check(self):
        policy = PPOPolicy(self.model, self.variant, deterministic=True)
        summary = summarize(evaluate(policy, self.episodes, self.max_decisions, self.solutions))
        row = {"decisions": self.num_timesteps, "minutes": round((time.time() - self.start_time) / 60, 2), **summary}
        new_file = not self.curve_path.exists()
        with open(self.curve_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        if summary["regret_m"] < self.best_regret:
            self.best_regret = summary["regret_m"]
            self.model.save(self.out_dir / "best_model.zip")
        if self.verbose:
            print(f"{self.num_timesteps:>9} decisions  regret {summary['regret_m']:.3f} m  "
                  f"success {summary['success']:.3f}  best {self.best_regret:.3f} m")
        self.next_check += self.every

    def _on_training_start(self):
        self._check()  # point 0 of the learning curve: the untrained policy

    def _on_step(self):
        if self.num_timesteps >= self.next_check:
            self._check()
        return True


def train(variant, seed, kappa=1.0, lam=DEFAULT_LAMBDA, placement="table", houses="pool",
          total_decisions=TOTAL_DECISIONS, eval_every=EVAL_EVERY, validation_episodes=VALIDATION_EPISODES,
          out_root="runs", verbose=1):
    torch.set_num_threads(1)  # the network is tiny; one thread is faster and lets runs share the CPU
    config = EpisodeConfig(rho=0.0, kappa=kappa, lam=lam, placement=placement,
                           max_decisions=DEFAULT_MAX_DECISIONS)  # rho is redrawn every reset
    train_houses = load_houses(houses, "train")
    val_houses = load_houses(houses, "validation")

    name = run_name(variant.name.lower(), seed, kappa, lam, placement)
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_curve.csv").unlink(missing_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "variant": variant.value, "seed": seed, "kappa": kappa, "lam": lam, "placement": placement,
        "houses": houses, "train_rho_range": TRAIN_RHO_RANGE, "max_decisions": config.max_decisions,
        "total_decisions": total_decisions, "eval_every": eval_every,
        "validation_episodes": validation_episodes, "validation_seed": VALIDATION_SEED,
        "ppo": PPO_SETTINGS, "net_arch": POLICY_KWARGS["net_arch"], "num_envs": NUM_ENVS,
    }, indent=2))

    vec_env = make_vec_env(
        lambda: TrustMapEnv(train_houses, config, variant, rho_range=TRAIN_RHO_RANGE),
        n_envs=NUM_ENVS, seed=seed, vec_env_cls=DummyVecEnv,
    )
    model = MaskablePPO("MlpPolicy", vec_env, policy_kwargs=POLICY_KWARGS, seed=seed,
                        device="cpu", verbose=0, **PPO_SETTINGS)
    # Fixed validation set; rho drawn per episode from the training range (Q7).
    val_eps = make_episodes(val_houses, config, validation_episodes, VALIDATION_SEED, rho_range=TRAIN_RHO_RANGE)
    callback = ValidationCallback(val_eps, config.max_decisions, variant, eval_every, out_dir)
    callback.verbose = verbose
    start = time.time()
    model.learn(total_timesteps=total_decisions, callback=callback)
    model.save(out_dir / "final_model.zip")
    minutes = (time.time() - start) / 60
    if verbose:
        print(f"done in {minutes:.1f} min; best validation regret {callback.best_regret:.3f} m; saved to {out_dir}")
    return out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    p.add_argument("--placement", choices=("table", "uniform"), default="table")
    p.add_argument("--houses", choices=("pool", "smoke"), default="pool",
                   help="'pool' = generated house pools (build once: python -m trustmap.pool); "
                        "'smoke' = stand-in houses for debugging only")
    p.add_argument("--total-decisions", type=int, default=TOTAL_DECISIONS)
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    p.add_argument("--validation-episodes", type=int, default=VALIDATION_EPISODES)
    p.add_argument("--out", default="runs")
    a = p.parse_args()
    train(VARIANTS[a.variant], a.seed, a.kappa, a.lam, a.placement, a.houses,
          a.total_decisions, a.eval_every, a.validation_episodes, a.out)


if __name__ == "__main__":
    main()
