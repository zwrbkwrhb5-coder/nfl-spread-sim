from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SPREAD_BASE = [
    "diff_off_epa",
    "diff_def_epa",
    "diff_off_success",
    "diff_def_success",
]

SPREAD_QB = [
    "diff_qb_epa_per_dropback",
    "diff_qb_success_rate",
    "diff_qb_cpoe",
    "diff_qb_sack_rate",
    "diff_qb_interception_rate",
    "diff_qb_explosive_pass_rate",
]

TOTAL_BASE = [
    "sum_off_epa",
    "sum_def_epa",
    "sum_off_success",
    "sum_def_success",
    "sum_points_for",
    "sum_points_against",
    "home_pre_adj_off_epa",
    "away_pre_adj_off_epa",
    "home_pre_adj_def_epa",
    "away_pre_adj_def_epa",
    "home_pre_explosive_pass_rate",
    "away_pre_explosive_pass_rate",
    "home_pre_turnover_rate",
    "away_pre_turnover_rate",
]

TOTAL_QB = [
    "home_pre_qb_epa_per_dropback",
    "away_pre_qb_epa_per_dropback",
    "home_pre_qb_success_rate",
    "away_pre_qb_success_rate",
    "home_pre_qb_cpoe",
    "away_pre_qb_cpoe",
    "home_pre_qb_sack_rate",
    "away_pre_qb_sack_rate",
    "home_pre_qb_interception_rate",
    "away_pre_qb_interception_rate",
    "home_pre_qb_explosive_pass_rate",
    "away_pre_qb_explosive_pass_rate",
]


def walk_forward_common_sample(
    df: pd.DataFrame,
    base_features: list[str],
    extra_features: list[str],
    target: str,
    first_test_season: int,
    alpha: float,
) -> dict:
    rows = []

    required = base_features + extra_features + [target]
    common = df.dropna(subset=required).copy()

    for season in sorted(
        int(s) for s in common["season"].unique()
        if int(s) >= first_test_season
    ):
        train = common[common["season"] < season]
        test = common[common["season"] == season]
        if train.empty or test.empty:
            continue

        season_result = {"season": season, "games": int(len(test))}

        for label, features in [
            ("base", base_features),
            ("plus_qb", base_features + extra_features),
        ]:
            model = Pipeline([
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ])
            model.fit(train[features], train[target])
            pred = model.predict(test[features])

            season_result[f"{label}_mae"] = float(
                mean_absolute_error(test[target], pred)
            )
            season_result[f"{label}_rmse"] = float(
                np.sqrt(mean_squared_error(test[target], pred))
            )

        rows.append(season_result)

    if not rows:
        raise RuntimeError("No common-sample walk-forward seasons available.")

    r = pd.DataFrame(rows)
    w = r["games"] / r["games"].sum()

    def weighted(c):
        return float((r[c] * w).sum())

    out = {
        "games": int(r["games"].sum()),
        "base_mae": weighted("base_mae"),
        "qb_mae": weighted("plus_qb_mae"),
        "delta_mae_qb_minus_base": weighted("plus_qb_mae") - weighted("base_mae"),
        "base_rmse": weighted("base_rmse"),
        "qb_rmse": weighted("plus_qb_rmse"),
        "delta_rmse_qb_minus_base": weighted("plus_qb_rmse") - weighted("base_rmse"),
        "by_season": rows,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--training-csv",
        default="artifacts/dual_market/training_games_qb_v17.csv",
    )
    ap.add_argument("--first-test-season", type=int, default=2022)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument(
        "--output",
        default="artifacts/dual_market/qb_ablation_v17.json",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.training_csv)

    spread = walk_forward_common_sample(
        df,
        SPREAD_BASE,
        SPREAD_QB,
        "actual_margin",
        args.first_test_season,
        args.alpha,
    )

    total = walk_forward_common_sample(
        df,
        TOTAL_BASE,
        TOTAL_QB,
        "actual_total",
        args.first_test_season,
        args.alpha,
    )

    report = {
        "spread": spread,
        "total": total,
        "interpretation": {
            "negative_delta_mae": "QB layer improved MAE",
            "negative_delta_rmse": "QB layer improved RMSE",
            "common_sample": True,
            "no_random_split": True,
        },
    }

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
