"""Point 4: observation rows and variant masks."""

import numpy as np
import pytest

from conftest import BATHROOM, KITCHEN, LIVING
from trustmap.episode import Episode
from trustmap.observation import FEATURES, Variant, build_observation


def make_episode(row_order):
    habit = np.array([1, 0.5, 8, 2, 2]) / 13.5  # towel, two-bedroom house
    return Episode(target="towel", r=0.55, c=0.55, claim_drawn_uniform=True,
                   claimed_room=LIVING, start_room=KITCHEN, claim_correct=False,
                   object_room=BATHROOM, habit=habit, row_order=np.array(row_order))


def test_rows_follow_shuffled_order(two_bedroom_house):
    ep = make_episode([2, 0, 4, 1, 3])
    searched = [True, False, False, False, False]
    obs = build_observation(two_bedroom_house, ep, KITCHEN, searched)
    assert obs.shape == (5, 5)
    # Row 3 is the living room: distance 4 / max 9, not searched, claimed, c, habit.
    assert obs[3] == pytest.approx([4 / 9, 0, 1, 0.55, 0.5 / 13.5])
    # Row 1 is the kitchen, the start room: distance 0, searched, not claimed.
    assert obs[1] == pytest.approx([0, 1, 0, 0, 1 / 13.5])
    assert obs[:, 0].max() <= 1.0
    assert obs[:, 2].sum() == 1 and obs[:, 3].sum() == pytest.approx(0.55)


def test_variant_masks(two_bedroom_house):
    ep = make_episode([0, 1, 2, 3, 4])
    searched = [True, False, False, False, False]
    full = build_observation(two_bedroom_house, ep, KITCHEN, searched, Variant.CONF)
    binary = build_observation(two_bedroom_house, ep, KITCHEN, searched, Variant.BINARY)
    noprior = build_observation(two_bedroom_house, ep, KITCHEN, searched, Variant.NOPRIOR)
    c, k = FEATURES.index("confidence"), FEATURES.index("claimed")
    assert full[:, c].sum() > 0 and full[:, k].sum() == 1
    assert binary[:, c].sum() == 0 and binary[:, k].sum() == 1
    assert noprior[:, c].sum() == 0 and noprior[:, k].sum() == 0
    # Everything else is identical across variants.
    keep = [i for i in range(5) if i not in (c, k)]
    assert np.array_equal(full[:, keep], noprior[:, keep])
