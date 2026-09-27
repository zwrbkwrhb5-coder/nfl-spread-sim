from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def new_model(random_state: int = 7):
    return HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=350,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        min_samples_leaf=20,
        random_state=random_state,
    )

def walk_forward_predictions(df: pd.DataFrame, features: list[str], first_test_season: int):
    preds = []
    for test_season in sorted(s for s in df["season"].unique() if s >= first_test_season):
        train = df[df["season"] < test_season]
        test = df[df["season"] == test_season]
        if train.empty or test.empty:
            continue

        m = new_model()
        m.fit(train[features], train["home_margin"])
        p = m.predict(test[features])

        part = test[["game_id", "season", "week", "home_team", "away_team", "home_margin"]].copy()
        part["pred_margin"] = p
        preds.append(part)

    out = pd.concat(preds, ignore_index=True)
    out["residual"] = out["home_margin"] - out["pred_margin"]
    return out

def summarize_validation(preds: pd.DataFrame) -> dict:
    y = preds["home_margin"].to_numpy()
    p = preds["pred_margin"].to_numpy()
    return {
        "games": int(len(preds)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "bias": float(np.mean(p - y)),
        "residual_std": float(preds["residual"].std(ddof=1)),
    }
