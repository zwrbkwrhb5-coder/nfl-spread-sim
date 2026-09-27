import pandas as pd

from score_model.market_stack_v37 import (
    frozen_season_predictions,
    validate_no_future,
)


def make_data():
    rows = []
    for season in [2022, 2023, 2024]:
        for week in [1, 2]:
            for i in range(120):
                market_margin = (i % 7) - 3
                pred_margin = market_margin + 1.0
                total_line = 44 + (i % 3)
                pred_total = total_line + 2.0

                rows.append({
                    "game_id": f"{season}_{week}_{i}",
                    "season": season,
                    "week": week,
                    "week_key": season * 100 + week,
                    "actual_margin": market_margin + 0.5,
                    "actual_total": total_line + 1.0,
                    "pred_margin": pred_margin,
                    "pred_total": pred_total,
                    "market_margin": market_margin,
                    "total_line": total_line,
                })
    return pd.DataFrame(rows)


def test_frozen_training_is_prior_seasons_only():
    d = make_data()
    pred, coef = frozen_season_predictions(
        d,
        first_eval_season=2023,
        min_prior_rows=200,
    )

    audit = validate_no_future(coef)
    assert audit["all_training_seasons_strictly_prior"] is True

    c2023 = coef[coef["target_season"] == 2023]
    assert (c2023["source_max_season"] == 2022).all()


def test_predictions_created():
    d = make_data()
    pred, _ = frozen_season_predictions(
        d,
        first_eval_season=2023,
        min_prior_rows=200,
    )

    assert pred["spread_stacked"].notna().all()
    assert pred["total_stacked"].notna().all()
