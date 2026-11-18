"""PPO settings and the wrapper that lets a trained agent play like a baseline (Point 7).

PPO (Proximal Policy Optimization) in plain words: a neural network outputs
one score per room; masked rooms get minus infinity; a softmax turns the
scores into probabilities; the room is sampled from them. After collecting a
batch of decisions, PPO nudges the network toward choices that turned out
better than expected (positive advantage) and away from worse ones, but clips
each update so the policy changes only a little at a time. A second network,
the value network, estimates the expected return from a state, like V(s).
"""

import torch
from torch import nn

from trustmap.observation import build_observation

# Point 7c: SB3 defaults, with the changes this problem needs.
PPO_SETTINGS = dict(
    gamma=1.0,          # 6c: no discounting (library default 0.99)
    gae_lambda=1.0,     # advantage = full Monte Carlo return minus V(s), as in M4-1 (default 0.95);
                        # PPO's own lambda, unrelated to our lambda = 0.5 of 3c
    ent_coef=0.01,      # entropy bonus: a small reward for keeping room probabilities spread out,
                        # a smooth version of epsilon-greedy exploration (default 0)
    learning_rate=3e-4, # step size of each network update
    n_steps=2048,       # decisions collected per environment copy before each update
    batch_size=64,      # minibatch size inside an update
    n_epochs=10,        # passes over each batch of collected decisions
    clip_range=0.2,     # limit on how far one update may move the policy
)

# Point 7b: flat network, two hidden layers of 64 units with tanh, for both the
# policy (5 outputs, one score per row) and the value network (1 output). These
# are SB3's MlpPolicy defaults, written out so they are visible.
POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=nn.Tanh)

NUM_ENVS = 8  # 7c: 8 copies of the 2D simulator stepping side by side (vectorized environments)


class PPOPolicy:
    """Lets a trained MaskablePPO model play through the same loop as the baselines.

    deterministic=True picks the highest-probability room instead of sampling
    (greedy actions at evaluation, Q7 answered 2026-11-18), so the reported
    metrics carry no sampling noise.
    """

    def __init__(self, model, variant, deterministic=True, name=None):
        self.model = model
        self.variant = variant
        self.deterministic = deterministic
        self.name = name or variant.value

    def choose(self, search, rng):
        obs = build_observation(search.house, search.episode, search.current_room, search.searched, self.variant)
        with torch.no_grad():
            action, _ = self.model.predict(
                obs.reshape(-1), action_masks=search.action_mask(), deterministic=self.deterministic
            )
        return int(action)
