import pandas as pd
import numpy as np
from score_model.market_backtest_v15 import (
    ensure_prediction_columns,
    spread_grade,
    total_grade,
    wilson_interval,
    bootstrap_roi_ci,
)

def test_derive_columns():
    df = pd.DataFrame([{
        "pred_home_score":27,"pred_away_score":21,
        "home_score":24,"away_score":20
    }])
    out = ensure_prediction_columns(df)
    assert out.loc[0,"pred_margin"] == 6
    assert out.loc[0,"pred_total"] == 48
    assert out.loc[0,"actual_margin"] == 4
    assert out.loc[0,"actual_total"] == 44

def test_spread_grades():
    assert spread_grade(7,-3.5) == "home"
    assert spread_grade(2,-3.5) == "away"
    assert spread_grade(3,-3) == "push"

def test_total_grades():
    assert total_grade(48,45.5) == "over"
    assert total_grade(42,45.5) == "under"
    assert total_grade(45,45) == "push"

def test_wilson_bounds():
    lo,hi = wilson_interval(55,100)
    assert 0 <= lo < hi <= 1

def test_bootstrap_roi_ci():
    lo,hi = bootstrap_roi_ci(np.array([1,-1,1,-1,.9,-1,1,.9]))
    assert lo <= hi
