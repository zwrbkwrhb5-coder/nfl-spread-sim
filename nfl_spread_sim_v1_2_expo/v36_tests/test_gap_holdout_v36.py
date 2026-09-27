import pandas as pd

from score_model.gap_holdout_v36 import (
    THRESHOLD,
    american_profit,
    wilson_interval,
)


def test_threshold_is_frozen_at_five():
    assert THRESHOLD == 5.0


def test_american_profit():
    assert abs(american_profit(-110) - (100/110)) < 1e-12
    assert american_profit(150) == 1.5


def test_wilson_bounds():
    lo, hi = wilson_interval(6, 10)
    assert 0 <= lo <= 0.6
    assert 0.6 <= hi <= 1
