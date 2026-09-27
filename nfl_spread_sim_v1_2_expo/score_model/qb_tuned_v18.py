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
    "diff_off_epa","diff_def_epa","diff_off_success","diff_def_success"
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
    "sum_off_epa","sum_def_epa","sum_off_success","sum_def_success",
    "sum_points_for","sum_points_against",
    "home_pre_adj_off_epa","away_pre_adj_off_epa",
    "home_pre_adj_def_epa","away_pre_adj_def_epa",
    "home_pre_explosive_pass_rate","away_pre_explosive_pass_rate",
    "home_pre_turnover_rate","away_pre_turnover_rate",
]

TOTAL_QB = [
    "home_pre_qb_epa_per_dropback","away_pre_qb_epa_per_dropback",
    "home_pre_qb_success_rate","away_pre_qb_success_rate",
    "home_pre_qb_cpoe","away_pre_qb_cpoe",
    "home_pre_qb_sack_rate","away_pre_qb_sack_rate",
    "home_pre_qb_interception_rate","away_pre_qb_interception_rate",
    "home_pre_qb_explosive_pass_rate","away_pre_qb_explosive_pass_rate",
]


def shrink_qb_features(df: pd.DataFrame, scale_dropbacks: float = 300.0) -> pd.DataFrame:
    d = df.copy()
    for side in ["home","away"]:
        w = d[f"{side}_qb_prior_dropbacks"].fillna(0) / (
            d[f"{side}_qb_prior_dropbacks"].fillna(0) + scale_dropbacks
        )
        cols = [
            "qb_epa_per_dropback","qb_success_rate","qb_cpoe",
            "qb_sack_rate","qb_interception_rate","qb_explosive_pass_rate",
        ]
        for c in cols:
            col = f"{side}_pre_{c}"
            if col not in d:
                continue
            league = d[col].mean(skipna=True)
            d[col] = league + w * (d[col] - league)

    pairs = [
        ("qb_epa_per_dropback","diff_qb_epa_per_dropback"),
        ("qb_success_rate","diff_qb_success_rate"),
        ("qb_cpoe","diff_qb_cpoe"),
        ("qb_sack_rate","diff_qb_sack_rate"),
        ("qb_interception_rate","diff_qb_interception_rate"),
        ("qb_explosive_pass_rate","diff_qb_explosive_pass_rate"),
    ]
    for base,out in pairs:
        d[out] = d[f"home_pre_{base}"] - d[f"away_pre_{base}"]
    return d


def evaluate(df, features, target, first_test_season=2022, alpha=1.0):
    rows = []
    for season in sorted(s for s in df["season"].unique() if s >= first_test_season):
        tr = df[df["season"] < season].dropna(subset=features+[target])
        te = df[df["season"] == season].dropna(subset=features+[target])
        if tr.empty or te.empty:
            continue
        model = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        model.fit(tr[features], tr[target])
        pred = model.predict(te[features])
        rows.append({
            "season": int(season),
            "games": int(len(te)),
            "mae": float(mean_absolute_error(te[target], pred)),
            "rmse": float(np.sqrt(mean_squared_error(te[target], pred))),
        })
    r = pd.DataFrame(rows)
    w = r["games"] / r["games"].sum()
    return {
        "games": int(r["games"].sum()),
        "mae": float((r["mae"]*w).sum()),
        "rmse": float((r["rmse"]*w).sum()),
        "by_season": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", default="artifacts/dual_market/training_games_qb_v17.csv")
    ap.add_argument("--output", default="artifacts/dual_market/qb_tuning_v18.json")
    ap.add_argument("--first-test-season", type=int, default=2022)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    df = pd.read_csv(args.training_csv)

    scales = [100, 200, 300, 500, 800]
    rows = []
    for scale in scales:
        d = shrink_qb_features(df, scale_dropbacks=float(scale))

        s_base = evaluate(d, SPREAD_BASE, "actual_margin", args.first_test_season, args.alpha)
        s_qb = evaluate(d, SPREAD_BASE+SPREAD_QB, "actual_margin", args.first_test_season, args.alpha)
        t_base = evaluate(d, TOTAL_BASE, "actual_total", args.first_test_season, args.alpha)
        t_qb = evaluate(d, TOTAL_BASE+TOTAL_QB, "actual_total", args.first_test_season, args.alpha)

        rows.append({
            "shrinkage_dropbacks": scale,
            "spread_delta_mae": s_qb["mae"] - s_base["mae"],
            "spread_delta_rmse": s_qb["rmse"] - s_base["rmse"],
            "total_delta_mae": t_qb["mae"] - t_base["mae"],
            "total_delta_rmse": t_qb["rmse"] - t_base["rmse"],
            "spread_base": s_base,
            "spread_qb": s_qb,
            "total_base": t_base,
            "total_qb": t_qb,
        })

    best_spread = min(rows, key=lambda x: (x["spread_delta_mae"], x["spread_delta_rmse"]))
    best_total = min(rows, key=lambda x: (x["total_delta_mae"], x["total_delta_rmse"]))

    out = {
        "tested_shrinkage_scales": scales,
        "best_spread": best_spread,
        "best_total": best_total,
        "all_results": rows,
    }

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
