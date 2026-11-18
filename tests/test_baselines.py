"""Point 8b regret and the Point 8d baselines."""

import numpy as np
import pytest

from conftest import BATHROOM, BEDROOM_A, KITCHEN, LIVING, random_house
from tests.test_observation import make_episode
from trustmap.baselines import ALL_BASELINES, BayesNominal, GreedyMapFollower, NearestUnexplored, Random
from trustmap.episode import EpisodeConfig, generate_episode
from trustmap.metrics import episode_metrics, optimum_for
from trustmap.search import RoomSearch


def play(policy, house, episode, rng, max_decisions=3):
    s = RoomSearch(house, episode, max_decisions)
    while not s.done:
        row = policy.choose(s, rng)
        assert s.action_mask()[row], "baseline picked a masked room"
        s.step(row)
    return s


def test_nearest_and_follower_orders(two_bedroom_house):
    ep = make_episode([2, 0, 4, 1, 3])  # start kitchen, claim living, object bathroom
    rng = np.random.default_rng(0)
    s = play(NearestUnexplored(), two_bedroom_house, ep, rng)
    assert [h[2] for h in s.history] == [LIVING, BATHROOM]   # kitchen->living 4 m, living->bathroom 5 m
    s = play(GreedyMapFollower(), two_bedroom_house, ep, rng)
    assert s.history[0][2] == LIVING


def test_regret_zero_for_optimal_and_positive_otherwise(two_bedroom_house):
    ep = make_episode([2, 0, 4, 1, 3])  # r = 0.55 on living
    rng = np.random.default_rng(0)
    s = play(BayesNominal(), two_bedroom_house, ep, rng)  # c = r here, so this is the optimum
    m = episode_metrics(s, optimum_for(s))
    assert m.regret_m == pytest.approx(0.0, abs=1e-12)

    s = RoomSearch(two_bedroom_house, ep, 3)
    sol = optimum_for(s)
    q = sol.q_values(KITCHEN, 1 << KITCHEN)
    s.step(s.row_of_room(BEDROOM_A))  # a poor first choice
    if not s.done:
        while not s.done:
            s.step(s.row_of_room(sol.best_room(s.current_room, s.searched_bits)))
    m = episode_metrics(s, sol)
    assert m.regret_m == pytest.approx((q.max() - q[BEDROOM_A]) * 9.0)
    assert m.regret_m > 0


def test_all_baselines_respect_mask_and_cap_and_regret_is_nonnegative():
    rng = np.random.default_rng(1)
    for _ in range(300):
        house = random_house(rng)
        ep = generate_episode(house, EpisodeConfig(rho=float(rng.uniform(0, 0.8)), kappa=3.0), rng)
        for cls in ALL_BASELINES:
            s = play(cls(), house, ep, rng)
            assert s.decisions <= 3
            assert episode_metrics(s, optimum_for(s)).regret_m >= -1e-9


def test_bayes_nominal_equals_optimum_when_calibrated():
    rng = np.random.default_rng(2)
    for _ in range(200):
        house = random_house(rng)
        ep = generate_episode(house, EpisodeConfig(rho=0.4, kappa=1.0), rng)
        s = play(BayesNominal(), house, ep, rng)
        assert episode_metrics(s, optimum_for(s)).regret_m == pytest.approx(0.0, abs=1e-9)


def test_random_baseline_spreads_first_choice(two_bedroom_house):
    ep = make_episode([2, 0, 4, 1, 3])
    rng = np.random.default_rng(3)
    firsts = [play(Random(), two_bedroom_house, ep, rng).history[0][2] for _ in range(400)]
    assert set(firsts) == {LIVING, BATHROOM, BEDROOM_A, 4}
