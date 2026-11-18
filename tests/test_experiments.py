"""Experiment machinery (Point 9): records, c*, t-intervals. Uses the saved test pool."""

import numpy as np
import pytest

from trustmap import experiments as ex
from trustmap.pool import pool_path

pytestmark = pytest.mark.skipif(not pool_path("test").exists(), reason="build the pools first: python -m trustmap.pool")


def test_t_interval():
    mean, half = ex.t_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean == 3.0 and half == pytest.approx(2.776 * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5))
    assert ex.t_interval([2.5]) == (2.5, 0.0)


def test_records_for_baselines_and_optimum(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "RESULTS_DIR", tmp_path)
    cond = ex.Condition("main")
    ex.evaluate_condition("t", cond, (0.4,), [*ex.BASELINES, ex.OPTIMUM], with_c_star=True,
                          workers=1, per_house=1)
    rec = ex.load_records("t", cond, 0.4)
    assert len(rec.episode["c"]) == 200 and rec.episode["house_seed"][0] == 200_000
    assert rec.per_seed("optimum", "regret_m")[0] == pytest.approx(0.0, abs=1e-9)
    assert rec.per_seed("bayes_nominal", "regret_m")[0] == pytest.approx(0.0, abs=1e-9)  # kappa = 1
    assert rec.per_seed("follower", "followed")[0] == 1.0
    assert rec.per_seed("random", "regret_m")[0] > rec.per_seed("follower", "regret_m")[0]
    # The optimum follows the claim exactly when c >= c* (kappa = 1, so c = r).
    has = ~np.isnan(rec.episode["c_star"])
    followed = rec.runs[("optimum", None)]["followed"].astype(bool)
    assert (followed[has] == (rec.episode["c"][has] >= rec.episode["c_star"][has])).mean() > 0.99
