"""The numpy copy of a trained policy picks the same rooms as the SB3 model (Point 9e runner)."""

import numpy as np
import pytest

pytest.importorskip("sb3_contrib")

from conftest import random_house  # noqa: E402
from trustmap.env import TrustMapEnv  # noqa: E402
from trustmap.episode import EpisodeConfig, generate_episode  # noqa: E402
from trustmap.observation import Variant  # noqa: E402
from trustmap.policy_export import NumpyPolicy, export_policy  # noqa: E402
from trustmap.ppo import POLICY_KWARGS, PPOPolicy  # noqa: E402
from trustmap.search import RoomSearch  # noqa: E402


def test_numpy_policy_matches_sb3(tmp_path):
    from sb3_contrib import MaskablePPO

    rng = np.random.default_rng(0)
    houses = [random_house(rng) for _ in range(10)]
    env = TrustMapEnv(houses, EpisodeConfig(rho=0.4, kappa=1.0), Variant.BINARY)
    # An untrained network with random weights is enough to compare the two forward passes.
    model = MaskablePPO("MlpPolicy", env, policy_kwargs=POLICY_KWARGS, seed=3, device="cpu")
    model.save(tmp_path / "m.zip")
    export_policy(tmp_path / "m.zip", tmp_path / "m.npz", "binary")
    torch_policy = PPOPolicy(MaskablePPO.load(tmp_path / "m.zip", device="cpu"), Variant.BINARY)
    numpy_policy = NumpyPolicy(tmp_path / "m.npz")
    assert numpy_policy.variant is Variant.BINARY
    for _ in range(500):
        house = houses[rng.integers(len(houses))]
        ep = generate_episode(house, EpisodeConfig(rho=float(rng.uniform(0, 0.8)), kappa=1.0), rng)
        s = RoomSearch(house, ep, 3)
        while not s.done:
            row = torch_policy.choose(s, rng)
            assert numpy_policy.choose(s, rng) == row
            s.step(row)
