import pandas as pd

from score_model.anchor_robustness_v35 import (
    frozen_season_anchor,
    validate_frozen,
)


def make_data():
    rows = []

    for season in [2022, 2023, 2024]:
        for week in [1, 2]:
            for g in range(120):
                rows.append({
                    "game_id": f"{season}_{week}_{g}",
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

    return pd.DataFrame(rows)


def test_frozen_season_uses_prior_seasons_only():
    d = make_data()

    out, weights = frozen_season_anchor(
        d,
        first_eval_season=2023,
        min_prior_rows=200,
    )

    audit = validate_frozen(out)

    assert audit[
        "all_anchor_source_seasons_strictly_prior"
    ] is True

    w2023 = weights[
        weights["target_season"] == 2023
    ].iloc[0]

    assert w2023["source_max_season"] == 2022


def test_weight_is_frozen_within_target_season():
    d = make_data()

    out, _ = frozen_season_anchor(
        d,
        first_eval_season=2023,
        min_prior_rows=200,
    )

    for season, g in out.groupby("season"):
        assert g["frozen_spread_weight"].nunique() == 1
        assert g["frozen_total_weight"].nunique() == 1
