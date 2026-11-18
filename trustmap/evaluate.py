"""Compare trained PPO agents with the Point 8d baselines on held-out houses.

Run, for example (the 20 November go/no-go check: PPO-Conf against random):

    python -m trustmap.evaluate --model runs/conf_kappa1_lam0.5_table_seed0/best_model.zip --variant conf

For each rho, one fixed list of test episodes is drawn and every method plays
exactly those episodes (paired evaluation). The optimum (8a) is the regret-0
reference line, so it is not listed.

This is the quick check used during development (default 200 episodes per
rho). The reported E1-E3 results come from `python -m trustmap.e1` (and e2, e3),
which use the full Point 9b test set of 5,000 episodes per rho.
"""

import argparse

from trustmap.baselines import ALL_BASELINES
from trustmap.episode import DEFAULT_LAMBDA, DEFAULT_MAX_DECISIONS, EpisodeConfig
from trustmap.evaluation import evaluate, make_episodes, solve_all, summarize
from trustmap.train import VARIANTS, load_houses

TEST_SEED = 200_000


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", default=[], help="MaskablePPO .zip (repeat for several)")
    p.add_argument("--variant", action="append", default=[], choices=VARIANTS,
                   help="variant of each --model, in the same order")
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    p.add_argument("--placement", choices=("table", "uniform"), default="table")
    p.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=TEST_SEED)
    p.add_argument("--houses", choices=("pool", "smoke"), default="pool")
    a = p.parse_args()
    if len(a.model) != len(a.variant):
        p.error("give one --variant per --model")

    policies = [cls() for cls in ALL_BASELINES]
    if a.model:
        import torch
        from sb3_contrib import MaskablePPO

        from trustmap.ppo import PPOPolicy

        torch.set_num_threads(1)
        for path, v in zip(a.model, a.variant):
            policies.append(PPOPolicy(MaskablePPO.load(path, device="cpu"), VARIANTS[v], name=f"{VARIANTS[v].value} ({path})"))

    houses = load_houses(a.houses, "test")
    if a.houses == "smoke":
        print("WARNING: stand-in smoke houses; these numbers are for debugging only.\n")
    header = f"{'method':<60}{'regret m':>10}{'success':>9}{'SPL':>7}{'1st':>7}{'dist m':>8}"
    for rho in a.rho:
        cfg = EpisodeConfig(rho=rho, kappa=a.kappa, lam=a.lam, placement=a.placement,
                            max_decisions=DEFAULT_MAX_DECISIONS)
        episodes = make_episodes(houses, cfg, a.episodes, seed=a.seed)
        solutions = solve_all(episodes, cfg.max_decisions)
        print(f"rho = {rho:g}, kappa = {a.kappa:g}, {a.episodes} episodes")
        print(header)
        for policy in policies:
            s = summarize(evaluate(policy, episodes, cfg.max_decisions, solutions, seed=a.seed))
            print(f"{policy.name:<60}{s['regret_m']:>10.3f}{s['success']:>9.3f}{s['spl']:>7.3f}"
                  f"{s['success_first_decision']:>7.3f}{s['distance_m']:>8.2f}")
        print()


if __name__ == "__main__":
    main()
