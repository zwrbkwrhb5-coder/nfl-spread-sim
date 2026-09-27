import numpy as np
import pandas as pd

from score_model.market_anchor_v34 import (
    learn_weight,
    prepare,
    walk_forward_anchor,
    validate_time_safety,
)


def test_weight_zero_when_gap_has_no_signal():
    x = np.array([1.0, -1.0, 2.0, -2.0])
    y = np.array([-1.0, 1.0, -2.0, 2.0])
    assert learn_weight(x, y) == 0.0


def test_weight_one_when_model_gap_is_exact():
    x = np.array([1.0, -2.0, 3.0])
    y = x.copy()
    assert abs(learn_weight(x, y) - 1.0) < 1e-12


def test_spread_sign_convention():
    predictions = pd.DataFrame([{
        "game_id": "g",
        "season": 2022,
        "week": 1,
        "actual_margin": 7.0,
        "actual_total": 44.0,
        "pred_margin": 5.0,
        "pred_total": 45.0,
    }])

    market = pd.DataFrame([{
        "game_id": "g",
        "home_spread": -3.0,
        "total_line": 43.0,
    }])

    d = prepare(predictions, market)

    assert d.iloc[0]["market_margin"] == 3.0
    assert d.iloc[0]["spread_model_gap"] == 2.0
    assert d.iloc[0]["spread_actual_market_residual"] == 4.0


def test_walk_forward_is_prior_only():
    rows = []

    for season in [2022, 2023]:
        for week in range(1, 4):
            rows.append({
                "game_id": f"{season}_{week}",
                "season": season,
                "week": week,
                "week_key": season * 100 + week,
                "actual_margin": 3.0,
                "actual_total": 44.0,
                "pred_margin": 4.0,
                "pred_total": 45.0,
                "home_spread": -2.0,
                "total_line": 43.0,
                "market_margin": 2.0,
                "spread_model_gap": 2.0,
                "spread_actual_market_residual": 1.0,
                "total_model_gap": 2.0,
                "total_actual_market_residual": 1.0,
            })

    d = pd.DataFrame(rows)

    out = walk_forward_anchor(
        d,
        min_prior_rows=2,
        first_eval_season=2023,
    )

    audit = validate_time_safety(out)
    assert audit["spread_all_sources_strictly_prior"]
    assert audit["total_all_sources_strictly_prior"]
