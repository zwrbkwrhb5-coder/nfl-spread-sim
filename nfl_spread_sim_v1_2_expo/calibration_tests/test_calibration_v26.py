import numpy as np
import pandas as pd

from score_model.calibration_selection_v26 import (
    american_break_even,
    grade_value,
    raw_probability_from_residuals,
    build_raw_rows,
)


def test_break_even_minus_110():
    assert abs(american_break_even(-110) - 0.5238095238) < 1e-8


def test_grade_value():
    assert grade_value(1.0) == "win"
    assert grade_value(-1.0) == "loss"
    assert grade_value(0.0) == "push"


def test_empirical_probability():
    residuals = np.array([-2, -1, 0, 1, 2], dtype=float)
    assert raw_probability_from_residuals(residuals, 0, "gt") == 0.4


def test_same_week_is_not_used_in_residual_bank():
    market = pd.DataFrame([
        {
            "game_id":"2022_01_A_B","season":2022,"week":1,
            "home_spread":-3.0,"total_line":45.0,
            "home_spread_odds":-110,"away_spread_odds":-110,
            "over_odds":-110,"under_odds":-110,
            "week_key":202201,
        },
        {
            "game_id":"2022_02_C_D","season":2022,"week":2,
            "home_spread":-3.0,"total_line":45.0,
            "home_spread_odds":-110,"away_spread_odds":-110,
            "over_odds":-110,"under_odds":-110,
            "week_key":202202,
        },
        {
            "game_id":"2022_02_E_F","season":2022,"week":2,
            "home_spread":-3.0,"total_line":45.0,
            "home_spread_odds":-110,"away_spread_odds":-110,
            "over_odds":-110,"under_odds":-110,
            "week_key":202202,
        },
    ])

    pred = pd.DataFrame([
        {
            "game_id":"2022_01_A_B","season":2022,"week":1,"week_key":202201,
            "pred_margin":0.0,"pred_total":45.0,
            "actual_margin":10.0,"actual_total":55.0,
        },
        {
            "game_id":"2022_02_C_D","season":2022,"week":2,"week_key":202202,
            "pred_margin":0.0,"pred_total":45.0,
            "actual_margin":100.0,"actual_total":100.0,
        },
        {
            "game_id":"2022_02_E_F","season":2022,"week":2,"week_key":202202,
            "pred_margin":0.0,"pred_total":45.0,
            "actual_margin":-100.0,"actual_total":0.0,
        },
    ])

    rows = build_raw_rows(market, pred, min_prior_games=1)

    # Both week-2 games must get identical prior residual-bank probabilities;
    # neither can use the other week-2 final result.
    w2 = rows[
        (rows["week"] == 2)
        & (rows["market"] == "spread")
        & (rows["side"] == "home")
    ]
    assert len(w2) == 2
    assert w2["raw_probability"].nunique() == 1
