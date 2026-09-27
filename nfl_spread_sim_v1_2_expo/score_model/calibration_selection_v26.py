from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


DEFAULT_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
DEFAULT_LAMBDAS = [0.25, 0.50, 0.75, 1.00]


def american_break_even(odds: float) -> float:
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


def american_profit(odds: float) -> float:
    odds = float(odds)
    if odds < 0:
        return 100.0 / (-odds)
    return odds / 100.0


def ensure_market_columns(m: pd.DataFrame) -> pd.DataFrame:
    d = m.copy()

    if "home_spread" not in d.columns:
        if "spread_line" in d.columns:
            # nflverse convention: positive spread_line means home favored.
            d["home_spread"] = -pd.to_numeric(d["spread_line"], errors="coerce")
        else:
            raise ValueError("Market CSV needs home_spread or spread_line.")

    if "away_spread_odds" not in d.columns:
        d["away_spread_odds"] = -110
    if "home_spread_odds" not in d.columns:
        d["home_spread_odds"] = -110
    if "over_odds" not in d.columns:
        d["over_odds"] = -110
    if "under_odds" not in d.columns:
        d["under_odds"] = -110

    required = ["game_id", "season", "week", "home_spread", "total_line"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Market CSV missing required columns: {missing}")

    d["week_key"] = (
        pd.to_numeric(d["season"], errors="coerce").astype(int) * 100
        + pd.to_numeric(d["week"], errors="coerce").astype(int)
    )
    return d


def ensure_prediction_columns(p: pd.DataFrame) -> pd.DataFrame:
    d = p.copy()

    if "pred_margin" not in d.columns:
        d["pred_margin"] = d["pred_home_score"] - d["pred_away_score"]
    if "pred_total" not in d.columns:
        d["pred_total"] = d["pred_home_score"] + d["pred_away_score"]

    if "actual_margin" not in d.columns:
        d["actual_margin"] = d["home_score"] - d["away_score"]
    if "actual_total" not in d.columns:
        d["actual_total"] = d["home_score"] + d["away_score"]

    required = [
        "game_id", "season", "week",
        "pred_margin", "pred_total",
        "actual_margin", "actual_total",
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Prediction CSV missing required columns: {missing}")

    d["week_key"] = (
        pd.to_numeric(d["season"], errors="coerce").astype(int) * 100
        + pd.to_numeric(d["week"], errors="coerce").astype(int)
    )
    return d


def raw_probability_from_residuals(residuals, threshold, direction="gt"):
    x = np.asarray(residuals, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    if direction == "gt":
        return float(np.mean(x > threshold))
    return float(np.mean(x < threshold))


def grade_value(value: float, tol: float = 1e-12) -> str:
    if value > tol:
        return "win"
    if value < -tol:
        return "loss"
    return "push"


def result_profit(result: str, odds: float) -> float:
    if result == "win":
        return american_profit(odds)
    if result == "loss":
        return -1.0
    return 0.0


def build_raw_rows(
    market: pd.DataFrame,
    pred: pd.DataFrame,
    min_prior_games: int = 50,
) -> pd.DataFrame:
    """
    Build strictly prior-WEEK raw probabilities.

    Residual banks use only rows with week_key < current week_key.
    No same-week result can enter a current game's probability.
    """
    # Avoid duplicate actual_margin / actual_total columns from market data.
    market_for_merge = market.drop(
        columns=["actual_margin", "actual_total"],
        errors="ignore",
    )

    merged = market_for_merge.merge(
        pred[
            [
                "game_id", "season", "week", "week_key",
                "pred_margin", "pred_total",
                "actual_margin", "actual_total",
            ]
        ],
        on=["game_id", "season", "week", "week_key"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["week_key", "game_id"]).reset_index(drop=True)

    rows = []

    for _, r in merged.iterrows():
        prior = merged[merged["week_key"] < r["week_key"]]

        mres = (
            prior["actual_margin"] - prior["pred_margin"]
        ).dropna().to_numpy(float)
        tres = (
            prior["actual_total"] - prior["pred_total"]
        ).dropna().to_numpy(float)

        if len(mres) < min_prior_games or len(tres) < min_prior_games:
            continue

        # Spread:
        # actual_margin = pred_margin + residual
        # home covers when actual_margin + home_spread > 0.
        home_threshold = -(float(r["pred_margin"]) + float(r["home_spread"]))
        p_home = raw_probability_from_residuals(mres, home_threshold, "gt")
        p_away = 1.0 - p_home

        # Totals:
        # actual_total = pred_total + residual
        # over when residual > total_line - pred_total.
        total_threshold = float(r["total_line"]) - float(r["pred_total"])
        p_over = raw_probability_from_residuals(tres, total_threshold, "gt")
        p_under = 1.0 - p_over

        actual_margin = float(r["actual_margin"])
        actual_total = float(r["actual_total"])
        home_spread = float(r["home_spread"])
        total_line = float(r["total_line"])

        options = [
            {
                "market": "spread",
                "side": "home",
                "raw_probability": p_home,
                "odds": float(r["home_spread_odds"]),
                "grade_value": actual_margin + home_spread,
            },
            {
                "market": "spread",
                "side": "away",
                "raw_probability": p_away,
                "odds": float(r["away_spread_odds"]),
                "grade_value": -(actual_margin + home_spread),
            },
            {
                "market": "total",
                "side": "over",
                "raw_probability": p_over,
                "odds": float(r["over_odds"]),
                "grade_value": actual_total - total_line,
            },
            {
                "market": "total",
                "side": "under",
                "raw_probability": p_under,
                "odds": float(r["under_odds"]),
                "grade_value": total_line - actual_total,
            },
        ]

        for o in options:
            result = grade_value(o["grade_value"])
            rows.append({
                "game_id": r["game_id"],
                "season": int(r["season"]),
                "week": int(r["week"]),
                "week_key": int(r["week_key"]),
                "market": o["market"],
                "side": o["side"],
                "raw_probability": float(o["raw_probability"]),
                "odds": float(o["odds"]),
                "break_even": american_break_even(o["odds"]),
                "result": result,
                "win": np.nan if result == "push" else float(result == "win"),
                "profit": result_profit(result, o["odds"]),
                "pred_margin": float(r["pred_margin"]),
                "pred_total": float(r["pred_total"]),
                "home_spread": home_spread,
                "total_line": total_line,
            })

    return pd.DataFrame(rows)


def add_time_safe_isotonic(
    rows: pd.DataFrame,
    min_calibration_rows: int = 100,
) -> pd.DataFrame:
    """
    Calibrate each row using only prior-WEEK settled rows from the same market.
    """
    d = rows.sort_values(["week_key", "game_id", "market", "side"]).copy()
    out = []

    for _, r in d.iterrows():
        prior = d[
            (d["week_key"] < r["week_key"])
            & (d["market"] == r["market"])
            & d["win"].notna()
        ].copy()

        p = float(r["raw_probability"])

        if len(prior) >= min_calibration_rows and prior["win"].nunique() >= 2:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(
                prior["raw_probability"].to_numpy(float),
                prior["win"].to_numpy(float),
            )
            p_cal = float(iso.predict([p])[0])
        else:
            p_cal = p

        item = r.to_dict()
        item["calibrated_probability"] = p_cal
        item["calibration_rows"] = int(len(prior))
        out.append(item)

    return pd.DataFrame(out)


def brier_score(prob, win):
    p = np.asarray(prob, dtype=float)
    y = np.asarray(win, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean((p[mask] - y[mask]) ** 2))


def choose_lambda_and_threshold(
    prior: pd.DataFrame,
    thresholds,
    lambdas,
    min_bets_for_threshold: int = 75,
):
    settled = prior[prior["win"].notna()].copy()

    if len(settled) < min_bets_for_threshold:
        return 0.50, 0.03, "fallback"

    # Choose probability shrinkage by prior Brier score.
    lambda_scores = []
    for lam in lambdas:
        p = 0.5 + float(lam) * (
            settled["calibrated_probability"].to_numpy(float) - 0.5
        )
        lambda_scores.append(
            (brier_score(p, settled["win"]), float(lam))
        )

    lambda_scores.sort(key=lambda x: x[0])
    best_lambda = lambda_scores[0][1]

    # Choose minimum edge using only prior rows.
    tmp = settled.copy()
    tmp["p_selected"] = 0.5 + best_lambda * (
        tmp["calibrated_probability"] - 0.5
    )
    tmp["edge_selected"] = tmp["p_selected"] - tmp["break_even"]

    threshold_scores = []
    for threshold in thresholds:
        bets = tmp[tmp["edge_selected"] >= float(threshold)]
        if len(bets) < min_bets_for_threshold:
            continue
        roi = float(bets["profit"].sum() / len(bets))
        # Tie-break toward more conservative threshold, then more bets.
        threshold_scores.append(
            (roi, float(threshold), int(len(bets)))
        )

    if not threshold_scores:
        return best_lambda, 0.03, "lambda_only"

    threshold_scores.sort(
        key=lambda x: (x[0], x[1], x[2]),
        reverse=True,
    )
    return best_lambda, threshold_scores[0][1], "tuned"


def apply_time_safe_selection(
    rows: pd.DataFrame,
    thresholds=None,
    lambdas=None,
    min_bets_for_threshold: int = 75,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if lambdas is None:
        lambdas = DEFAULT_LAMBDAS

    d = rows.sort_values(
        ["week_key", "game_id", "market", "side"]
    ).copy()

    selected_rows = []

    # Tune once per week/market from strictly earlier weeks.
    for week_key in sorted(d["week_key"].unique()):
        week = d[d["week_key"] == week_key]

        for market_name in ["spread", "total"]:
            current = week[week["market"] == market_name].copy()
            if current.empty:
                continue

            prior = d[
                (d["week_key"] < week_key)
                & (d["market"] == market_name)
            ].copy()

            lam, threshold, mode = choose_lambda_and_threshold(
                prior,
                thresholds,
                lambdas,
                min_bets_for_threshold=min_bets_for_threshold,
            )

            current["selected_lambda"] = lam
            current["selected_threshold"] = threshold
            current["selection_mode"] = mode
            current["selected_probability"] = 0.5 + lam * (
                current["calibrated_probability"] - 0.5
            )
            current["selected_edge"] = (
                current["selected_probability"] - current["break_even"]
            )

            # Best side for each game/market only.
            current = (
                current.sort_values(
                    ["game_id", "selected_edge"],
                    ascending=[True, False],
                )
                .drop_duplicates(["game_id", "market"])
            )

            current["bet"] = (
                current["selected_edge"] >= current["selected_threshold"]
            )

            selected_rows.append(current)

    if not selected_rows:
        return pd.DataFrame()

    return pd.concat(selected_rows, ignore_index=True)


def summarize_bets(d: pd.DataFrame):
    bets = d[d["bet"]].copy()
    settled = bets[bets["result"] != "push"].copy()

    if len(bets) == 0:
        return {
            "bets": 0,
            "settled": 0,
            "pushes": 0,
            "win_rate": None,
            "units": 0.0,
            "roi": None,
            "brier": None,
            "avg_selected_edge": None,
        }

    return {
        "bets": int(len(bets)),
        "settled": int(len(settled)),
        "pushes": int((bets["result"] == "push").sum()),
        "win_rate": (
            float((settled["result"] == "win").mean())
            if len(settled) else None
        ),
        "units": float(bets["profit"].sum()),
        "roi": float(bets["profit"].sum() / len(bets)),
        "brier": brier_score(
            settled["selected_probability"],
            settled["win"],
        ) if len(settled) else None,
        "avg_selected_edge": float(bets["selected_edge"].mean()),
    }


def summarize_raw_best_side(d: pd.DataFrame):
    x = d.copy()
    x["selected_probability"] = x["raw_probability"]
    x["selected_edge"] = x["raw_probability"] - x["break_even"]
    x["selected_threshold"] = 0.0

    x = (
        x.sort_values(
            ["game_id", "market", "selected_edge"],
            ascending=[True, True, False],
        )
        .drop_duplicates(["game_id", "market"])
    )
    x["bet"] = x["selected_edge"] >= 0.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--market-csv",
        default="data/historical_market.csv",
    )
    ap.add_argument(
        "--predictions",
        default="artifacts/dual_market/oos_qb_total_context_v25.csv",
    )
    ap.add_argument(
        "--output-dir",
        default="artifacts/calibration_v26",
    )
    ap.add_argument("--min-prior-games", type=int, default=50)
    ap.add_argument("--min-calibration-rows", type=int, default=100)
    ap.add_argument("--min-bets-for-threshold", type=int, default=75)
    args = ap.parse_args()

    market = ensure_market_columns(pd.read_csv(args.market_csv))
    pred = ensure_prediction_columns(pd.read_csv(args.predictions))

    raw = build_raw_rows(
        market,
        pred,
        min_prior_games=args.min_prior_games,
    )

    calibrated = add_time_safe_isotonic(
        raw,
        min_calibration_rows=args.min_calibration_rows,
    )

    selected = apply_time_safe_selection(
        calibrated,
        min_bets_for_threshold=args.min_bets_for_threshold,
    )

    raw_best = summarize_raw_best_side(raw)

    report = {
        "validation": {
            "prior_week_residuals_only": True,
            "same_week_results_used": False,
            "isotonic_uses_prior_weeks_only": True,
            "threshold_tuning_uses_prior_weeks_only": True,
            "probability_shrinkage_tuned_on_prior_weeks_only": True,
        },
        "raw_no_threshold": {
            "spread": summarize_bets(
                raw_best[raw_best["market"] == "spread"]
            ),
            "total": summarize_bets(
                raw_best[raw_best["market"] == "total"]
            ),
        },
        "calibrated_selected": {
            "spread": summarize_bets(
                selected[selected["market"] == "spread"]
            ),
            "total": summarize_bets(
                selected[selected["market"] == "total"]
            ),
        },
    }

    # Add season diagnostics.
    season_rows = []
    for season in sorted(selected["season"].unique()):
        for market_name in ["spread", "total"]:
            ss = selected[
                (selected["season"] == season)
                & (selected["market"] == market_name)
            ]
            summary = summarize_bets(ss)
            season_rows.append({
                "season": int(season),
                "market": market_name,
                **summary,
            })

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw.to_csv(outdir / "raw_sides.csv", index=False)
    calibrated.to_csv(outdir / "calibrated_sides.csv", index=False)
    selected.to_csv(outdir / "selected_rows.csv", index=False)
    pd.DataFrame(season_rows).to_csv(
        outdir / "summary_by_season.csv",
        index=False,
    )
    (outdir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nSaved outputs to {outdir}")


if __name__ == "__main__":
    main()
