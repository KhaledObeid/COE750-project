"""Gymnasium environment (Point 7) and paired evaluation. Skipped if gymnasium is not installed."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from conftest import random_house  # noqa: E402
from trustmap.baselines import NearestUnexplored, Random  # noqa: E402
from trustmap.env import TrustMapEnv  # noqa: E402
from trustmap.episode import EpisodeConfig  # noqa: E402
from trustmap.evaluation import evaluate, make_episodes, solve_all, summarize  # noqa: E402
from trustmap.observation import Variant  # noqa: E402


@pytest.fixture
def houses():
    rng = np.random.default_rng(0)
    return [random_house(rng) for _ in range(20)]


def test_env_passes_gymnasium_checker(houses):
    from gymnasium.utils.env_checker import check_env

    env = TrustMapEnv(houses, EpisodeConfig(rho=0.4, kappa=1.0))
    check_env(env, skip_render_check=True)


def test_episode_ends_within_cap_and_mask_shrinks(houses):
    env = TrustMapEnv(houses, EpisodeConfig(rho=0.4, kappa=1.0), rho_range=(0.0, 0.8))
    for seed in range(200):
        obs, _ = env.reset(seed=seed)
        assert obs.shape == (25,)
        mask = env.action_masks()
        assert mask.sum() == 4  # start room searched
        total, steps, done = 0.0, 0, False
        while not done:
            action = int(np.flatnonzero(env.action_masks())[0])
            obs, reward, done, truncated, info = env.step(action)
            total += reward
            steps += 1
            assert env.action_masks().sum() == 4 - steps or done
        assert steps <= 3 and not truncated
        assert info["success"] or total < -1.0  # failure includes the -1 penalty


def test_same_seed_same_episode(houses):
    a = TrustMapEnv(houses, EpisodeConfig(rho=0.4, kappa=1.0))
    b = TrustMapEnv(houses, EpisodeConfig(rho=0.4, kappa=1.0))
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)
    assert np.array_equal(oa, ob)


def test_variant_zeroes_inputs(houses):
    env = TrustMapEnv(houses, EpisodeConfig(rho=0.4, kappa=1.0), variant=Variant.NOPRIOR)
    obs, _ = env.reset(seed=1)
    rows = obs.reshape(5, 5)
    assert (rows[:, 2] == 0).all() and (rows[:, 3] == 0).all()


def test_paired_episodes_identical_for_every_method(houses):
    cfg = EpisodeConfig(rho=0.4, kappa=1.0)
    e1 = make_episodes(houses, cfg, 50, seed=3)
    e2 = make_episodes(houses, cfg, 50, seed=3)
    assert [(h.seed, id(h), ep.object_room, ep.claimed_room, ep.r) for h, ep in e1] == \
           [(h.seed, id(h), ep.object_room, ep.claimed_room, ep.r) for h, ep in e2]
    sols = solve_all(e1, 3)
    rand = summarize(evaluate(Random(), e1, 3, sols))
    near = summarize(evaluate(NearestUnexplored(), e1, 3, sols))
    assert rand["episodes"] == near["episodes"] == 50
    assert rand["regret_m"] >= 0 and near["regret_m"] >= 0
