from __future__ import annotations

import argparse, json
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

SPREAD_OPENING = [
    "diff_opening_matchup_opening_points",
    "diff_opening_matchup_opening_epa",
    "diff_opening_matchup_opening_success_rate",
    "diff_opening_matchup_opening_turnover",
    "diff_opening_matchup_opening_scored",
    "diff_opening_matchup_opening_td",
]

TOTAL_OPENING = [
    "sum_opening_matchup_opening_points",
    "sum_opening_matchup_opening_epa",
    "sum_opening_matchup_opening_success_rate",
    "sum_opening_matchup_opening_turnover",
    "sum_opening_matchup_opening_scored",
    "sum_opening_matchup_opening_td",
]


def evaluate_common(df, base, extra, target, first_test_season=2022):
    common = df.dropna(subset=base + extra + [target]).copy()

    rows = []
    for season in sorted(s for s in common.season.unique() if s >= first_test_season):
        tr = common[common.season < season]
        te = common[common.season == season]
        if tr.empty or te.empty:
            continue

        row = {"season":int(season),"games":int(len(te))}
        for label, feats in [("base",base),("plus_opening",base+extra)]:
            m = Pipeline([
                ("scale",StandardScaler()),
                ("ridge",Ridge(alpha=1.0)),
            ])
            m.fit(tr[feats],tr[target])
            p = m.predict(te[feats])
            row[f"{label}_mae"] = float(mean_absolute_error(te[target],p))
            row[f"{label}_rmse"] = float(np.sqrt(mean_squared_error(te[target],p)))
        rows.append(row)

    r = pd.DataFrame(rows)
    w = r.games / r.games.sum()
    wm = lambda c: float((r[c]*w).sum())

    return {
        "games":int(r.games.sum()),
        "base_mae":wm("base_mae"),
        "opening_mae":wm("plus_opening_mae"),
        "delta_mae":wm("plus_opening_mae")-wm("base_mae"),
        "base_rmse":wm("base_rmse"),
        "opening_rmse":wm("plus_opening_rmse"),
        "delta_rmse":wm("plus_opening_rmse")-wm("base_rmse"),
        "by_season":rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", default="artifacts/dual_market/training_games_qb_opening_v22.csv")
    ap.add_argument("--first-test-season", type=int, default=2022)
    ap.add_argument("--output", default="artifacts/dual_market/opening_drive_ablation_v22.json")
    args = ap.parse_args()

    df = pd.read_csv(args.training_csv)

    report = {
        "spread": evaluate_common(
            df, SPREAD_BASE, SPREAD_OPENING, "actual_margin", args.first_test_season
        ),
        "total": evaluate_common(
            df, TOTAL_BASE, TOTAL_OPENING, "actual_total", args.first_test_season
        ),
        "notes": {
            "common_sample": True,
            "pregame_only": True,
            "same_game_opening_drive_used": False,
            "baseline": "QB-enhanced model",
        }
    }

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
