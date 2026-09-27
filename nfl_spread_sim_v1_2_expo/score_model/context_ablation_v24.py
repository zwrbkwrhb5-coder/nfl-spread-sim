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
    "diff_off_epa","diff_def_epa","diff_off_success","diff_def_success",
    "diff_qb_epa_per_dropback","diff_qb_success_rate","diff_qb_cpoe",
    "diff_qb_sack_rate","diff_qb_interception_rate","diff_qb_explosive_pass_rate",
]

TOTAL_BASE = [
    "sum_off_epa","sum_def_epa","sum_off_success","sum_def_success",
    "sum_points_for","sum_points_against",
    "home_pre_adj_off_epa","away_pre_adj_off_epa",
    "home_pre_adj_def_epa","away_pre_adj_def_epa",
    "home_pre_explosive_pass_rate","away_pre_explosive_pass_rate",
    "home_pre_turnover_rate","away_pre_turnover_rate",
    "home_pre_qb_epa_per_dropback","away_pre_qb_epa_per_dropback",
    "home_pre_qb_success_rate","away_pre_qb_success_rate",
    "home_pre_qb_cpoe","away_pre_qb_cpoe",
    "home_pre_qb_sack_rate","away_pre_qb_sack_rate",
    "home_pre_qb_interception_rate","away_pre_qb_interception_rate",
    "home_pre_qb_explosive_pass_rate","away_pre_qb_explosive_pass_rate",
]

SPREAD_CONTEXT = [
    "rest_diff",
    "short_week_diff",
    "long_rest_diff",
    "divisional_game",
    "indoor",
    "grass",
]

TOTAL_CONTEXT = [
    "divisional_game",
    "indoor",
    "grass",
    "cold_degrees",
    "wind_over_10",
    "cold_game",
    "windy_game",
]


def evaluate_common(df, base, extra, target, first_test_season=2022):
    common = df.dropna(subset=base + extra + [target]).copy()
    rows = []

    for season in sorted(int(s) for s in common["season"].unique() if int(s) >= first_test_season):
        train = common[common["season"] < season]
        test = common[common["season"] == season]
        if train.empty or test.empty:
            continue

        row = {"season": season, "games": int(len(test))}
        for label, features in [("base", base), ("plus_context", base + extra)]:
            model = Pipeline([
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ])
            model.fit(train[features], train[target])
            pred = model.predict(test[features])
            row[f"{label}_mae"] = float(mean_absolute_error(test[target], pred))
            row[f"{label}_rmse"] = float(np.sqrt(mean_squared_error(test[target], pred)))
        rows.append(row)

    r = pd.DataFrame(rows)
    if r.empty:
        raise RuntimeError("No walk-forward seasons available.")

    w = r["games"] / r["games"].sum()
    weighted = lambda c: float((r[c] * w).sum())

    return {
        "games": int(r["games"].sum()),
        "base_mae": weighted("base_mae"),
        "context_mae": weighted("plus_context_mae"),
        "delta_mae": weighted("plus_context_mae") - weighted("base_mae"),
        "base_rmse": weighted("base_rmse"),
        "context_rmse": weighted("plus_context_rmse"),
        "delta_rmse": weighted("plus_context_rmse") - weighted("base_rmse"),
        "by_season": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", default="artifacts/dual_market/training_games_qb_context_v24.csv")
    ap.add_argument("--first-test-season", type=int, default=2022)
    ap.add_argument("--output", default="artifacts/dual_market/context_ablation_v24.json")
    args = ap.parse_args()

    df = pd.read_csv(args.training_csv)

    report = {
        "spread": evaluate_common(df, SPREAD_BASE, SPREAD_CONTEXT, "actual_margin", args.first_test_season),
        "total": evaluate_common(df, TOTAL_BASE, TOTAL_CONTEXT, "actual_total", args.first_test_season),
        "notes": {
            "baseline": "QB-enhanced model",
            "common_sample": True,
            "random_split": False,
            "spread_context": SPREAD_CONTEXT,
            "total_context": TOTAL_CONTEXT,
        },
    }

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
