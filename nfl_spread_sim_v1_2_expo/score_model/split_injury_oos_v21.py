from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

QB_SPREAD_BASE = [
    "diff_off_epa","diff_def_epa","diff_off_success","diff_def_success",
    "diff_qb_epa_per_dropback","diff_qb_success_rate","diff_qb_cpoe",
    "diff_qb_sack_rate","diff_qb_interception_rate","diff_qb_explosive_pass_rate",
]

QB_TOTAL_BASE = [
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

SPREAD_PV_INJURY = [
    "diff_pv_total_injury_impact",
    "diff_pv_qb_injury_impact",
    "diff_pv_ol_injury_impact",
    "diff_pv_secondary_injury_impact",
    "diff_pv_skill_injury_impact",
]

TOTAL_BASIC_INJURY = [
    "sum_total_injury_impact",
    "sum_qb_injury_impact",
    "sum_ol_injury_impact",
    "sum_secondary_injury_impact",
    "sum_skill_injury_impact",
]


def fit_predict_walk_forward(df, spread_features, total_features, first_test_season, max_test_season):
    rows = []

    for season in sorted(
        int(s) for s in df["season"].unique()
        if int(s) >= first_test_season and int(s) <= max_test_season
    ):
        train = df[df["season"] < season].copy()
        test = df[df["season"] == season].copy()

        req = spread_features + total_features + ["actual_margin","actual_total"]
        test = test.dropna(subset=req)

        train_s = train.dropna(subset=spread_features + ["actual_margin"])
        train_t = train.dropna(subset=total_features + ["actual_total"])

        if train_s.empty or train_t.empty or test.empty:
            continue

        sm = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        tm = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])

        sm.fit(train_s[spread_features], train_s["actual_margin"])
        tm.fit(train_t[total_features], train_t["actual_total"])

        pred_margin = sm.predict(test[spread_features])
        pred_total = tm.predict(test[total_features])

        pred_home = (pred_total + pred_margin) / 2.0
        pred_away = (pred_total - pred_margin) / 2.0

        for i, (_, r) in enumerate(test.iterrows()):
            rows.append({
                "game_id": r["game_id"],
                "season": int(r["season"]),
                "week": int(r["week"]),
                "home_score": r["home_score"],
                "away_score": r["away_score"],
                "pred_margin": float(pred_margin[i]),
                "pred_total": float(pred_total[i]),
                "pred_home_score": float(pred_home[i]),
                "pred_away_score": float(pred_away[i]),
                "actual_margin": float(r["actual_margin"]),
                "actual_total": float(r["actual_total"]),
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--training-csv",
        default="artifacts/dual_market/training_games_qb_injury_pv_v20.csv",
    )
    ap.add_argument("--first-test-season", type=int, default=2022)
    ap.add_argument("--max-test-season", type=int, default=2024)
    ap.add_argument(
        "--qb-output",
        default="artifacts/dual_market/oos_qb_common_v21.csv",
    )
    ap.add_argument(
        "--split-output",
        default="artifacts/dual_market/oos_split_injury_v21.csv",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.training_csv)

    qb = fit_predict_walk_forward(
        df,
        QB_SPREAD_BASE,
        QB_TOTAL_BASE,
        args.first_test_season,
        args.max_test_season,
    )

    split = fit_predict_walk_forward(
        df,
        QB_SPREAD_BASE + SPREAD_PV_INJURY,
        QB_TOTAL_BASE + TOTAL_BASIC_INJURY,
        args.first_test_season,
        args.max_test_season,
    )

    # Restrict both to exact common game set.
    keys = ["game_id","season","week"]
    common = qb[keys].merge(split[keys], on=keys, how="inner").drop_duplicates()
    qb = qb.merge(common, on=keys, how="inner")
    split = split.merge(common, on=keys, how="inner")

    Path(args.qb_output).parent.mkdir(parents=True, exist_ok=True)
    qb.to_csv(args.qb_output, index=False)
    split.to_csv(args.split_output, index=False)

    print(f"QB baseline OOS: {args.qb_output} ({len(qb):,} games)")
    print(f"Split injury OOS: {args.split_output} ({len(split):,} games)")
    print(f"Common games: {len(common):,}")


if __name__ == "__main__":
    main()
