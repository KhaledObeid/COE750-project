"""Point 3c: episode generation recipe, checked against the worked numbers in DECISIONS.md."""

import numpy as np
import pytest

from conftest import BATHROOM, BEDROOM_A, BEDROOM_B, KITCHEN, LIVING
from trustmap.episode import EpisodeConfig, displayed_confidence, generate_episode, sample_reliability


def many(house, config, n, seed=0):
    rng = np.random.default_rng(seed)
    return [generate_episode(house, config, rng) for _ in range(n)]


def test_same_seed_same_episode(two_bedroom_house):
    cfg = EpisodeConfig(rho=0.4, kappa=1.0)
    a = generate_episode(two_bedroom_house, cfg, np.random.default_rng(7))
    b = generate_episode(two_bedroom_house, cfg, np.random.default_rng(7))
    assert (a.target, a.r, a.claimed_room, a.start_room, a.object_room) == (
        b.target, b.r, b.claimed_room, b.start_room, b.object_room)
    assert np.array_equal(a.row_order, b.row_order)


def test_beta_spread_at_rho_04():
    # DECISIONS 3c: Beta(6, 4), mean 0.6, standard deviation about 0.15.
    rng = np.random.default_rng(1)
    r = np.array([sample_reliability(0.4, rng) for _ in range(50_000)])
    assert r.mean() == pytest.approx(0.6, abs=0.005)
    assert r.std() == pytest.approx(0.148, abs=0.005)


def test_perfect_map_at_rho_zero(two_bedroom_house):
    for ep in many(two_bedroom_house, EpisodeConfig(rho=0.0, kappa=3.0), 200):
        assert ep.r == 1.0 and ep.c == 1.0 and ep.claim_correct


def test_confidence_formula():
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    logit = lambda p: np.log(p / (1 - p))
    for r in (0.1, 0.55, 0.9):
        for kappa in (0.5, 1.0, 3.0):
            assert displayed_confidence(r, kappa) == pytest.approx(sigmoid(kappa * logit(r)))
    assert displayed_confidence(0.55, 1.0) == pytest.approx(0.55)  # kappa = 1 is calibrated


def test_claim_distribution_towel_lambda_half(two_bedroom_house):
    # DECISIONS 3c: towel, two-bedroom house, lambda = 0.5:
    # bathroom 0.40, each bedroom 0.17, kitchen 0.14, living room 0.12.
    eps = many(two_bedroom_house, EpisodeConfig(rho=0.4, kappa=1.0, targets=("towel",)), 40_000)
    claims = np.bincount([e.claimed_room for e in eps], minlength=5) / len(eps)
    expected = {BATHROOM: 0.40, BEDROOM_A: 0.17, BEDROOM_B: 0.17, KITCHEN: 0.14, LIVING: 0.12}
    for room, p in expected.items():
        assert claims[room] == pytest.approx(p, abs=0.01)


def test_start_never_claimed_and_uniform(two_bedroom_house):
    eps = many(two_bedroom_house, EpisodeConfig(rho=0.4, kappa=1.0, lam=1.0), 20_000)
    assert all(e.start_room != e.claimed_room for e in eps)
    claimed_living = [e.start_room for e in eps if e.claimed_room == LIVING]
    share = np.bincount(claimed_living, minlength=5) / len(claimed_living)
    for room in (KITCHEN, BATHROOM, BEDROOM_A, BEDROOM_B):
        assert share[room] == pytest.approx(0.25, abs=0.03)


def test_claim_correct_with_probability_r(two_bedroom_house):
    eps = many(two_bedroom_house, EpisodeConfig(rho=0.4, kappa=1.0), 20_000)
    assert np.mean([e.claim_correct for e in eps]) == pytest.approx(np.mean([e.r for e in eps]), abs=0.01)
    for e in eps:
        assert (e.object_room == e.claimed_room) == e.claim_correct
        assert e.object_room != e.start_room


def test_wrong_claim_placement_worked_example(two_bedroom_house):
    # DECISIONS 3c recipe: towel, claim living, start kitchen, claim wrong ->
    # bathroom 8/12, each bedroom 2/12.
    cfg = EpisodeConfig(rho=0.6, kappa=1.0, lam=1.0, targets=("towel",))
    eps = [e for e in many(two_bedroom_house, cfg, 100_000)
           if e.claimed_room == LIVING and e.start_room == KITCHEN and not e.claim_correct]
    share = np.bincount([e.object_room for e in eps], minlength=5) / len(eps)
    assert share[BATHROOM] == pytest.approx(8 / 12, abs=0.03)
    assert share[BEDROOM_A] == pytest.approx(2 / 12, abs=0.03)
    assert share[BEDROOM_B] == pytest.approx(2 / 12, abs=0.03)


def test_uniform_placement_spreads_evenly(two_bedroom_house):
    cfg = EpisodeConfig(rho=0.6, kappa=1.0, lam=0.0, placement="uniform", targets=("towel",))
    eps = many(two_bedroom_house, cfg, 20_000)
    claims = np.bincount([e.claimed_room for e in eps], minlength=5) / len(eps)
    assert np.allclose(claims, 0.2, atol=0.015)
    assert np.allclose(eps[0].habit, 0.2)


def test_config_validation():
    with pytest.raises(ValueError):
        EpisodeConfig(rho=1.0, kappa=1.0)
    with pytest.raises(ValueError):
        EpisodeConfig(rho=0.4, kappa=1.0, placement="other")
    with pytest.raises(ValueError):
        EpisodeConfig(rho=0.4, kappa=1.0, targets=("spoon",))
