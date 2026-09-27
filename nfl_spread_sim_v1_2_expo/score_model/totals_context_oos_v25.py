from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SPREAD_FEATURES = [
    "diff_off_epa",
    "diff_def_epa",
    "diff_off_success",
    "diff_def_success",
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

TOTAL_CONTEXT = [
    "divisional_game",
    "indoor",
    "grass",
    "cold_degrees",
    "wind_over_10",
    "cold_game",
    "windy_game",
]


def make_model(alpha: float = 1.0):
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def build_walk_forward_predictions(
    df: pd.DataFrame,
    first_test_season: int = 2022,
    alpha: float = 1.0,
):
    """
    Produce two prediction tables on the exact same train/test rows:

    1. QB baseline spread + QB baseline total
    2. Identical spread prediction + QB/context total

    The common-sample restriction is applied BEFORE model fitting, so the
    baseline and context models receive the same historical training games
    and the same test games.
    """
    required = (
        SPREAD_FEATURES
        + TOTAL_BASE
        + TOTAL_CONTEXT
        + ["actual_margin", "actual_total", "home_score", "away_score"]
    )

    common = df.dropna(subset=required).copy()

    baseline_rows = []
    context_rows = []

    seasons = sorted(
        int(s)
        for s in common["season"].unique()
        if int(s) >= first_test_season
    )

    for season in seasons:
        train = common[common["season"] < season].copy()
        test = common[common["season"] == season].copy()

        if train.empty or test.empty:
            continue

        spread_model = make_model(alpha)
        total_base_model = make_model(alpha)
        total_context_model = make_model(alpha)

        spread_model.fit(train[SPREAD_FEATURES], train["actual_margin"])
        total_base_model.fit(train[TOTAL_BASE], train["actual_total"])
        total_context_model.fit(
            train[TOTAL_BASE + TOTAL_CONTEXT],
            train["actual_total"],
        )

        pred_margin = spread_model.predict(test[SPREAD_FEATURES])
        pred_total_base = total_base_model.predict(test[TOTAL_BASE])
        pred_total_context = total_context_model.predict(
            test[TOTAL_BASE + TOTAL_CONTEXT]
        )

        for i, (_, r) in enumerate(test.iterrows()):
            base_margin = float(pred_margin[i])
            base_total = float(pred_total_base[i])
            ctx_total = float(pred_total_context[i])

            baseline_rows.append({
                "game_id": r["game_id"],
                "season": int(r["season"]),
                "week": int(r["week"]),
                "home_score": float(r["home_score"]),
                "away_score": float(r["away_score"]),
                "pred_margin": base_margin,
                "pred_total": base_total,
                "pred_home_score": (base_total + base_margin) / 2.0,
                "pred_away_score": (base_total - base_margin) / 2.0,
                "actual_margin": float(r["actual_margin"]),
                "actual_total": float(r["actual_total"]),
            })

            context_rows.append({
                "game_id": r["game_id"],
                "season": int(r["season"]),
                "week": int(r["week"]),
                "home_score": float(r["home_score"]),
                "away_score": float(r["away_score"]),
                # Spread intentionally identical to baseline.
                "pred_margin": base_margin,
                "pred_total": ctx_total,
                "pred_home_score": (ctx_total + base_margin) / 2.0,
                "pred_away_score": (ctx_total - base_margin) / 2.0,
                "actual_margin": float(r["actual_margin"]),
                "actual_total": float(r["actual_total"]),
            })

    return pd.DataFrame(baseline_rows), pd.DataFrame(context_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--training-csv",
        default="artifacts/dual_market/training_games_qb_context_v24.csv",
    )
    ap.add_argument("--first-test-season", type=int, default=2022)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument(
        "--baseline-output",
        default="artifacts/dual_market/oos_qb_total_base_v25.csv",
    )
    ap.add_argument(
        "--context-output",
        default="artifacts/dual_market/oos_qb_total_context_v25.csv",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.training_csv)

    baseline, context = build_walk_forward_predictions(
        df,
        first_test_season=args.first_test_season,
        alpha=args.alpha,
    )

    if baseline.empty or context.empty:
        raise RuntimeError("No OOS predictions were generated.")

    if len(baseline) != len(context):
        raise RuntimeError("Baseline/context OOS row counts differ.")

    keys = ["game_id", "season", "week"]
    if not baseline[keys].equals(context[keys]):
        raise RuntimeError("Baseline/context game sets are not identical.")

    # Spread predictions must be exactly identical by construction.
    if not baseline["pred_margin"].equals(context["pred_margin"]):
        raise RuntimeError("Spread predictions changed in totals-context test.")

    Path(args.baseline_output).parent.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(args.baseline_output, index=False)
    context.to_csv(args.context_output, index=False)

    print(f"Saved baseline OOS: {args.baseline_output} ({len(baseline):,} games)")
    print(f"Saved context OOS:  {args.context_output} ({len(context):,} games)")
    print("Common sample: true")
    print("Spread predictions identical: true")


if __name__ == "__main__":
    main()
