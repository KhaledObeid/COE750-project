"""Run a trained PPO policy with numpy only (for the Gazebo runner, Point 9e).

ROS 2 Humble's Python is 3.10 without PyTorch, so the Gazebo runner cannot load
the SB3 model directly. The policy network is small enough to write out by
hand, which also shows exactly what the trained agent computes:

    h1     = tanh(W1 x + b1)          x = the 25 observation numbers (7b)
    h2     = tanh(W2 h1 + b2)
    scores = Wa h2 + ba               one score (logit) per observation row
    masked rows (already searched, 5a) get score -infinity
    chosen row = the row with the highest score (greedy action, Q7)

Export once, in the training environment. For the Gazebo test (9e), the
PPO-Conf seed with the median best validation regret (not the luckiest one):
    python -m trustmap.policy_export --median conf --out gazebo/policies/ppo_conf.npz
or any single checkpoint:
    python -m trustmap.policy_export --model runs/<run>/best_model.zip --variant conf --out x.npz
The test tests/test_policy_export.py checks that the numpy policy picks the
same room as the SB3 model on thousands of observations.
"""

import argparse

import numpy as np

from trustmap.observation import VARIANT_KEYS, build_observation

_LAYERS = {
    "w1": "mlp_extractor.policy_net.0.weight", "b1": "mlp_extractor.policy_net.0.bias",
    "w2": "mlp_extractor.policy_net.2.weight", "b2": "mlp_extractor.policy_net.2.bias",
    "wa": "action_net.weight", "ba": "action_net.bias",
}


def export_policy(model_path, out_path, variant_key):
    """Write the policy-network weights of a MaskablePPO checkpoint to a .npz file."""
    from sb3_contrib import MaskablePPO

    state = MaskablePPO.load(model_path, device="cpu").policy.state_dict()
    arrays = {k: state[name].numpy().astype(np.float64) for k, name in _LAYERS.items()}
    np.savez(out_path, variant=variant_key, source=str(model_path), **arrays)


class NumpyPolicy:
    """Greedy PPO policy from exported weights; same interface as the baselines."""

    def __init__(self, path, name=None):
        data = np.load(path)
        self.w = {k: data[k] for k in _LAYERS}
        self.variant = VARIANT_KEYS[str(data["variant"])]
        self.name = name or self.variant.value

    def scores(self, obs):
        h = np.tanh(self.w["w1"] @ obs + self.w["b1"])
        h = np.tanh(self.w["w2"] @ h + self.w["b2"])
        return self.w["wa"] @ h + self.w["ba"]

    def choose(self, search, rng):
        obs = build_observation(search.house, search.episode, search.current_room, search.searched, self.variant)
        scores = self.scores(obs.reshape(-1).astype(np.float64))
        scores[~search.action_mask()] = -np.inf
        return int(np.argmax(scores))


def median_validation_run(variant_key, seeds=range(5), kappa=1.0, runs_dir=None):
    """Run folder whose best validation regret is the median over the seeds (9e).

    Only finished runs count. With 5 seeds this is the third best.
    """
    import csv
    from pathlib import Path

    from trustmap.train import run_name

    runs_dir = Path(runs_dir or Path(__file__).resolve().parent.parent / "runs")
    best = []
    for s in seeds:
        d = runs_dir / run_name(variant_key, s, kappa)
        if not (d / "final_model.zip").exists():
            continue
        with open(d / "validation_curve.csv") as f:
            best.append((min(float(r["regret_m"]) for r in csv.DictReader(f)), d))
    if not best:
        raise FileNotFoundError(f"no finished {variant_key} runs in {runs_dir}")
    best.sort()
    return best[(len(best) - 1) // 2][1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--median", choices=list(VARIANT_KEYS), help="export the median-validation seed of this variant")
    p.add_argument("--model", help="a single checkpoint .zip instead of --median")
    p.add_argument("--variant", choices=list(VARIANT_KEYS), help="variant of --model")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.median:
        run_dir = median_validation_run(a.median)
        print(f"median validation run: {run_dir.name}")
        export_policy(run_dir / "best_model.zip", a.out, a.median)
    elif a.model and a.variant:
        export_policy(a.model, a.out, a.variant)
    else:
        p.error("give --median, or --model with --variant")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

