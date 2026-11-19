"""Gymnasium environment for training (Point 7), built from the shared pieces.

One environment step is one decision (a semi-MDP, Agreements): the agent
picks one of the 5 observation rows, the robot goes to that room along the
planned path, and the room is checked. The rules live in `RoomSearch`
(Points 5, 6), the observation in `build_observation` (Point 4), the episode
draw in `generate_episode` (Point 3c). This class only connects them to the
Gymnasium interface that Stable-Baselines3 expects.

Action masking (5a, 7a): `action_masks()` returns the five yes/no values of
the rows that may be chosen. MaskablePPO calls it before every decision and
gives masked rows probability exactly zero.

The observation is given to the network as one flat list of 25 numbers
(7b: rows in shuffled order, 5 features each, row after row).
"""

import dataclasses

import gymnasium as gym
import numpy as np

from trustmap.episode import generate_episode
from trustmap.house import NUM_ROOMS
from trustmap.observation import NUM_FEATURES, Variant, build_observation
from trustmap.search import RoomSearch


class TrustMapEnv(gym.Env):
    """Room-level object search with a prior map.

    houses:    the house pool to draw from (step 1 of the 3c recipe).
    config:    an `EpisodeConfig` with the dials (rho, kappa, lam, placement,
               targets, decision cap).
    variant:   which inputs the agent sees (4c).
    rho_range: if given as (low, high), rho is drawn uniformly from it at
               every reset instead of using config.rho. Training uses
               (0, 0.8), fresh every episode (9a).
    """

    metadata = {"render_modes": []}

    def __init__(self, houses, config, variant=Variant.CONF, rho_range=None):
        super().__init__()
        if not houses:
            raise ValueError("the house pool is empty")
        self.houses = list(houses)
        self.config = config
        self.variant = variant
        self.rho_range = rho_range
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(NUM_ROOMS * NUM_FEATURES,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(NUM_ROOMS)
        self.search = None

    def _observation(self):
        s = self.search
        obs = build_observation(s.house, s.episode, s.current_room, s.searched, self.variant)
        return obs.reshape(-1)

    def _info(self):
        s = self.search
        return {"success": s.success, "distance_m": s.distance_m, "decisions": s.decisions}

    def reset(self, *, seed=None, options=None):
        # All randomness of an episode happens here, from Gymnasium's seeded
        # generator self.np_random (Agreements: randomness only at reset).
        super().reset(seed=seed)
        rng = self.np_random
        house = self.houses[rng.integers(len(self.houses))]  # 3c step 1
        config = self.config
        if self.rho_range is not None:
            config = dataclasses.replace(config, rho=float(rng.uniform(*self.rho_range)))
        episode = generate_episode(house, config, rng)       # 3c steps 2-6 and row shuffle
        self.search = RoomSearch(house, episode, config.max_decisions)
        return self._observation(), self._info()

    def step(self, action):
        reward = self.search.step(int(action))
        terminated = self.search.done  # success, or the decision cap reached (6b)
        return self._observation(), float(reward), terminated, False, self._info()

    def action_masks(self):
        return self.search.action_mask()
