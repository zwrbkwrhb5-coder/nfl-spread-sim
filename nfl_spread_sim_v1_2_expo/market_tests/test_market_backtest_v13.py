import numpy as np
import pandas as pd

from score_model.market_backtest import (
    american_break_even,
    american_profit_per_unit,
    grade_spread,
    grade_total,
    ProbabilityCalibrator,
    edge_bucket_summary,
)


def test_break_even_minus_110():
    assert abs(american_break_even(-110) - (110/210)) < 1e-12


def test_profit_minus_110():
    assert abs(american_profit_per_unit(-110) - (100/110)) < 1e-12


def test_profit_plus_120():
    assert abs(american_profit_per_unit(120) - 1.2) < 1e-12


def test_spread_home_cover():
    assert grade_spread(7, -3.5) == "home_cover"


def test_spread_away_cover():
    assert grade_spread(2, -3.5) == "away_cover"


def test_spread_push():
    assert grade_spread(3, -3) == "push"


def test_total_over():
    assert grade_total(48, 46.5) == "over"


def test_total_under():
    assert grade_total(44, 46.5) == "under"


def test_total_push():
    assert grade_total(47, 47) == "push"


def test_calibrator_identity_when_too_small():
    c = ProbabilityCalibrator().fit([.4, .6], [0, 1])
    x = c.predict([.45, .55])
    assert np.allclose(x, [.45, .55])


def test_calibrator_bounds():
    x = np.linspace(.1, .9, 50)
    y = (x > .5).astype(int)
    c = ProbabilityCalibrator().fit(x, y)
    p = c.predict([0, 1])
    assert (p >= 0).all() and (p <= 1).all()


def test_edge_buckets_runs():
    df = pd.DataFrame({
        "market":["spread_home"]*4,
        "probability_edge":[-.01,.005,.025,.08],
        "line_edge_points":[0,1,2,3],
        "is_push":[False]*4,
        "won":[0,1,1,1],
        "profit_units":[-1,.91,.91,.91],
    })
    out = edge_bucket_summary(df)
    assert not out.empty
