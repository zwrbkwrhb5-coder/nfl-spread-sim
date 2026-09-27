import pandas as pd

from score_model.frozen_calibration_validation_v33 import (
    choose_lambda_by_prior_brier,
    run_frozen_validation,
)


def test_lambda_chosen_without_future_rows():
    prior = pd.DataFrame({
        "win": [1, 0, 1, 0],
        "p": [0.6, 0.4, 0.6, 0.4],
    })

    lam, table = choose_lambda_by_prior_brier(prior, "p")
    assert lam in {0.25, 0.5, 0.75, 1.0}
    assert len(table) == 4


def test_first_season_is_not_evaluated():
    rows = []

    for season in [2022, 2023]:
        for game in range(3):
            for market in ["spread", "total"]:
                for side, win in [("a", 1.0), ("b", 0.0)]:
                    rows.append({
                        "game_id": f"{season}_{game}_{market}",
                        "season": season,
                        "week": 1,
                        "week_key": season * 100 + 1,
                        "market": market,
                        "side": side,
                        "odds": -110,
                        "break_even": 110 / 210,
                        "result": "win" if win else "loss",
                        "win": win,
                        "profit": 100/110 if win else -1.0,
                        "raw_probability": 0.6 if win else 0.4,
                        "calibrated_probability": 0.58 if win else 0.42,
                        "platt_probability": 0.55 if win else 0.45,
                    })

    d = pd.DataFrame(rows)
    selected, lambdas = run_frozen_validation(d)

    assert set(selected["season"]) == {2023}
    assert set(lambdas["target_season"]) == {2023}
