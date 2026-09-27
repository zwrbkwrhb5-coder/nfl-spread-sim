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

QB_METRICS = [
    "qb_epa_per_dropback",
    "qb_success_rate",
    "qb_cpoe",
    "qb_sack_rate",
    "qb_interception_rate",
    "qb_explosive_pass_rate",
]

SPREAD_QB_RAW = [f"diff_{m}" for m in QB_METRICS]
SPREAD_QB_SAFE = [f"diff_{m}_v28" for m in QB_METRICS]

TOTAL_TEAM = [
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

TOTAL_QB_RAW = [
    f"{side}_pre_{m}"
    for side in ["home", "away"]
    for m in QB_METRICS
]

TOTAL_QB_SAFE = [
    f"{side}_pre_{m}_v28"
    for side in ["home", "away"]
    for m in QB_METRICS
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


def model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])


def metrics(y, p):
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
    }


def walk_forward_common(df: pd.DataFrame, first_test_season: int = 2022):
    spread_raw = SPREAD_BASE + SPREAD_QB_RAW
    spread_safe = SPREAD_BASE + SPREAD_QB_SAFE
    total_raw = TOTAL_TEAM + TOTAL_QB_RAW + TOTAL_CONTEXT
    total_safe = TOTAL_TEAM + TOTAL_QB_SAFE + TOTAL_CONTEXT

    required = sorted(set(
        spread_raw + spread_safe + total_raw + total_safe
        + ["actual_margin", "actual_total", "home_score", "away_score"]
    ))

    common = df.dropna(subset=required).copy()

    raw_rows = []
    safe_rows = []
    season_report = []

    for season in sorted(
        int(s) for s in common["season"].unique()
        if int(s) >= first_test_season
    ):
        tr = common[common["season"] < season].copy()
        te = common[common["season"] == season].copy()
        if tr.empty or te.empty:
            continue

        sr = model()
        ss = model()
        traw = model()
        tsafe = model()

        sr.fit(tr[spread_raw], tr["actual_margin"])
        ss.fit(tr[spread_safe], tr["actual_margin"])
        traw.fit(tr[total_raw], tr["actual_total"])
        tsafe.fit(tr[total_safe], tr["actual_total"])

        pm_raw = sr.predict(te[spread_raw])
        pm_safe = ss.predict(te[spread_safe])
        pt_raw = traw.predict(te[total_raw])
        pt_safe = tsafe.predict(te[total_safe])

        mr = metrics(te["actual_margin"], pm_raw)
        ms = metrics(te["actual_margin"], pm_safe)
        trm = metrics(te["actual_total"], pt_raw)
        tsm = metrics(te["actual_total"], pt_safe)

        season_report.append({
            "season": season,
            "games": int(len(te)),
            "spread_raw_mae": mr["mae"],
            "spread_safe_mae": ms["mae"],
            "spread_delta_mae": ms["mae"] - mr["mae"],
            "spread_raw_rmse": mr["rmse"],
            "spread_safe_rmse": ms["rmse"],
            "spread_delta_rmse": ms["rmse"] - mr["rmse"],
            "total_raw_mae": trm["mae"],
            "total_safe_mae": tsm["mae"],
            "total_delta_mae": tsm["mae"] - trm["mae"],
            "total_raw_rmse": trm["rmse"],
            "total_safe_rmse": tsm["rmse"],
            "total_delta_rmse": tsm["rmse"] - trm["rmse"],
        })

        for i, (_, r) in enumerate(te.iterrows()):
            base = {
                "game_id": r["game_id"],
                "season": int(r["season"]),
                "week": int(r["week"]),
                "home_score": float(r["home_score"]),
                "away_score": float(r["away_score"]),
                "actual_margin": float(r["actual_margin"]),
                "actual_total": float(r["actual_total"]),
            }

            raw_rows.append({
                **base,
                "pred_margin": float(pm_raw[i]),
                "pred_total": float(pt_raw[i]),
                "pred_home_score": float((pt_raw[i] + pm_raw[i]) / 2.0),
                "pred_away_score": float((pt_raw[i] - pm_raw[i]) / 2.0),
            })

            safe_rows.append({
                **base,
                "pred_margin": float(pm_safe[i]),
                "pred_total": float(pt_safe[i]),
                "pred_home_score": float((pt_safe[i] + pm_safe[i]) / 2.0),
                "pred_away_score": float((pt_safe[i] - pm_safe[i]) / 2.0),
            })

    raw = pd.DataFrame(raw_rows)
    safe = pd.DataFrame(safe_rows)
    by_season = pd.DataFrame(season_report)

    if raw.empty or safe.empty:
        raise RuntimeError("No OOS predictions generated.")

    if not raw[["game_id", "season", "week"]].equals(
        safe[["game_id", "season", "week"]]
    ):
        raise RuntimeError("Raw/safe OOS samples are not identical.")

    weights = by_season["games"] / by_season["games"].sum()
    weighted = lambda c: float((by_season[c] * weights).sum())

    report = {
        "common_sample_games": int(len(raw)),
        "common_sample": True,
        "spread": {
            "raw_mae": weighted("spread_raw_mae"),
            "safe_mae": weighted("spread_safe_mae"),
            "delta_mae_safe_minus_raw": weighted("spread_delta_mae"),
            "raw_rmse": weighted("spread_raw_rmse"),
            "safe_rmse": weighted("spread_safe_rmse"),
            "delta_rmse_safe_minus_raw": weighted("spread_delta_rmse"),
        },
        "total": {
            "raw_mae": weighted("total_raw_mae"),
            "safe_mae": weighted("total_safe_mae"),
            "delta_mae_safe_minus_raw": weighted("total_delta_mae"),
            "raw_rmse": weighted("total_raw_rmse"),
            "safe_rmse": weighted("total_safe_rmse"),
            "delta_rmse_safe_minus_raw": weighted("total_delta_rmse"),
        },
        "by_season": season_report,
    }

    return raw, safe, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--training-csv",
        default="artifacts/dual_market/training_games_qb_safe_context_v28.csv",
    )
    ap.add_argument("--first-test-season", type=int, default=2022)
    ap.add_argument(
        "--raw-output",
        default="artifacts/dual_market/oos_qb_raw_context_v28.csv",
    )
    ap.add_argument(
        "--safe-output",
        default="artifacts/dual_market/oos_qb_safe_context_v28.csv",
    )
    ap.add_argument(
        "--report-output",
        default="artifacts/dual_market/qb_safe_oos_v28.json",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.training_csv)
    raw, safe, report = walk_forward_common(
        df,
        first_test_season=args.first_test_season,
    )

    for path, data in [
        (args.raw_output, raw),
        (args.safe_output, safe),
    ]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(p, index=False)

    rp = Path(args.report_output)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved raw OOS:  {args.raw_output}")
    print(f"Saved safe OOS: {args.safe_output}")


if __name__ == "__main__":
    main()
