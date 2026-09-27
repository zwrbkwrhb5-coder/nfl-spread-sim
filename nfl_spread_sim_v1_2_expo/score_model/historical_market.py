from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .market_backtest import (
    ProbabilityCalibrator,
    american_break_even,
    edge_bucket_summary,
    grade_spread,
    grade_total,
    summarize_market,
    closing_line_summary,
)


REQUIRED_MARKET_COLUMNS = {
    "game_id", "season", "home_team", "away_team",
    "home_spread", "home_spread_odds",
    "total_line", "over_odds", "under_odds",
    "home_score", "away_score",
}


def load_market(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_MARKET_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"market file missing columns: {sorted(missing)}")
    return df.copy()


def load_oos_predictions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "game_id", "season",
        "pred_home_score", "pred_away_score",
        "pred_margin", "pred_total",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prediction file missing columns: {sorted(missing)}")
    return df.copy()


def build_residual_banks(oos: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    d = oos.dropna(subset=["actual_margin", "actual_total", "pred_margin", "pred_total"]).copy()
    margin_resid = (d["actual_margin"] - d["pred_margin"]).to_numpy(float)
    total_resid = (d["actual_total"] - d["pred_total"]).to_numpy(float)
    return margin_resid, total_resid


def empirical_probs(
    projection: float,
    residuals: np.ndarray,
    threshold: float,
    gt: bool,
) -> tuple[float, float]:
    sims = projection + residuals
    pushes = np.isclose(sims, threshold)
    wins = sims > threshold if gt else sims < threshold
    return float(wins.mean()), float(pushes.mean())


def build_rows(
    merged: pd.DataFrame,
    margin_resid: np.ndarray,
    total_resid: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for _, r in merged.iterrows():
        actual_margin = float(r["home_score"] - r["away_score"])
        actual_total = float(r["home_score"] + r["away_score"])

        # Home spread side
        home_cover_raw, spread_push_prob = empirical_probs(
            float(r["pred_margin"]),
            margin_resid,
            -float(r["home_spread"]),
            True,
        )
        spread_grade = grade_spread(actual_margin, float(r["home_spread"]))
        home_won = spread_grade == "home_cover"
        spread_push = spread_grade == "push"
        be_home = american_break_even(float(r["home_spread_odds"]))

        opening_line = r.get("opening_home_spread", np.nan)
        closing_line = r.get("closing_home_spread", np.nan)
        # Positive CLV means our chosen home side got a better number than close.
        clv = float(r["home_spread"] - closing_line) if pd.notna(closing_line) else np.nan

        rows.append({
            "game_id": r["game_id"],
            "season": int(r["season"]),
            "market": "spread_home",
            "selection": str(r["home_team"]),
            "line": float(r["home_spread"]),
            "odds": float(r["home_spread_odds"]),
            "closing_line": float(closing_line) if pd.notna(closing_line) else np.nan,
            "raw_probability": home_cover_raw,
            "push_probability": spread_push_prob,
            "break_even_probability": be_home,
            "probability_edge": home_cover_raw - be_home,
            "line_edge_points": float(r["pred_margin"] + r["home_spread"]),
            "won": int(home_won),
            "is_push": bool(spread_push),
            "actual_margin": actual_margin,
            "actual_total": actual_total,
            "clv_points": clv,
        })

        # Over side
        over_raw, total_push_prob = empirical_probs(
            float(r["pred_total"]),
            total_resid,
            float(r["total_line"]),
            True,
        )
        total_grade = grade_total(actual_total, float(r["total_line"]))
        over_won = total_grade == "over"
        total_push = total_grade == "push"
        be_over = american_break_even(float(r["over_odds"]))

        closing_total = r.get("closing_total", np.nan)
        over_clv = float(closing_total - r["total_line"]) if pd.notna(closing_total) else np.nan

        rows.append({
            "game_id": r["game_id"],
            "season": int(r["season"]),
            "market": "total_over",
            "selection": "OVER",
            "line": float(r["total_line"]),
            "odds": float(r["over_odds"]),
            "closing_line": float(closing_total) if pd.notna(closing_total) else np.nan,
            "raw_probability": over_raw,
            "push_probability": total_push_prob,
            "break_even_probability": be_over,
            "probability_edge": over_raw - be_over,
            "line_edge_points": float(r["pred_total"] - r["total_line"]),
            "won": int(over_won),
            "is_push": bool(total_push),
            "actual_margin": actual_margin,
            "actual_total": actual_total,
            "clv_points": over_clv,
        })

        # Under side
        under_raw, total_push_prob2 = empirical_probs(
            float(r["pred_total"]),
            total_resid,
            float(r["total_line"]),
            False,
        )
        under_won = total_grade == "under"
        be_under = american_break_even(float(r["under_odds"]))
        under_clv = float(r["total_line"] - closing_total) if pd.notna(closing_total) else np.nan

        rows.append({
            "game_id": r["game_id"],
            "season": int(r["season"]),
            "market": "total_under",
            "selection": "UNDER",
            "line": float(r["total_line"]),
            "odds": float(r["under_odds"]),
            "closing_line": float(closing_total) if pd.notna(closing_total) else np.nan,
            "raw_probability": under_raw,
            "push_probability": total_push_prob2,
            "break_even_probability": be_under,
            "probability_edge": under_raw - be_under,
            "line_edge_points": float(r["total_line"] - r["pred_total"]),
            "won": int(under_won),
            "is_push": bool(total_push),
            "actual_margin": actual_margin,
            "actual_total": actual_total,
            "clv_points": under_clv,
        })

    return pd.DataFrame(rows)


def walk_forward_calibrate(rows: pd.DataFrame, min_prior_rows: int = 100) -> pd.DataFrame:
    out = rows.copy()
    out["calibrated_probability"] = out["raw_probability"]

    for market in out["market"].unique():
        md = out[out["market"].eq(market)].copy()
        for season in sorted(md["season"].unique()):
            train = md[(md["season"] < season) & (~md["is_push"])].copy()
            test_idx = md[md["season"].eq(season)].index

            if len(train) < min_prior_rows:
                continue

            cal = ProbabilityCalibrator().fit(train["raw_probability"], train["won"])
            out.loc[test_idx, "calibrated_probability"] = cal.predict(
                out.loc[test_idx, "raw_probability"]
            )

    out["probability_edge"] = (
        out["calibrated_probability"] - out["break_even_probability"]
    )

    profit = []
    for _, r in out.iterrows():
        if bool(r["is_push"]):
            profit.append(0.0)
        elif int(r["won"]) == 1:
            odds = float(r["odds"])
            profit.append(odds / 100.0 if odds > 0 else 100.0 / (-odds))
        else:
            profit.append(-1.0)
    out["profit_units"] = profit
    return out


def choose_best_side(rows: pd.DataFrame) -> pd.DataFrame:
    # For spreads we currently only have a home-side quote in the required schema.
    # For totals, choose the stronger of Over vs Under for each game.
    spread = rows[rows["market"].eq("spread_home")].copy()

    totals = rows[rows["market"].isin(["total_over", "total_under"])].copy()
    totals = (
        totals.sort_values(
            ["game_id", "probability_edge", "line_edge_points"],
            ascending=[True, False, False]
        )
        .drop_duplicates("game_id")
    )
    return pd.concat([spread, totals], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-csv", required=True)
    ap.add_argument("--oos-predictions", required=True)
    ap.add_argument("--output-dir", default="artifacts/market_backtest_v13")
    ap.add_argument("--min-edge", type=float, default=0.0)
    args = ap.parse_args()

    market = load_market(args.market_csv)
    oos = load_oos_predictions(args.oos_predictions)

    if "actual_margin" not in oos:
        oos["actual_margin"] = oos["actual_home_score"] - oos["actual_away_score"]
    if "actual_total" not in oos:
        oos["actual_total"] = oos["actual_home_score"] + oos["actual_away_score"]

    margin_resid, total_resid = build_residual_banks(oos)

    merged = market.merge(
        oos[[
            "game_id", "season", "pred_home_score", "pred_away_score",
            "pred_margin", "pred_total"
        ]],
        on=["game_id", "season"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise RuntimeError("No games matched between market file and OOS predictions.")

    rows = build_rows(merged, margin_resid, total_resid)
    rows = walk_forward_calibrate(rows)
    best = choose_best_side(rows)
    bettable = best[best["probability_edge"] >= float(args.min_edge)].copy()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows.to_csv(out / "all_market_rows.csv", index=False)
    best.to_csv(out / "best_side_per_game.csv", index=False)
    bettable.to_csv(out / "bettable_rows.csv", index=False)
    edge_bucket_summary(best).to_csv(out / "edge_bucket_summary.csv", index=False)
    closing_line_summary(best).to_csv(out / "closing_line_summary.csv", index=False)

    report = {
        "matched_games": int(len(merged)),
        "market_rows": int(len(rows)),
        "selected_rows": int(len(best)),
        "minimum_probability_edge": float(args.min_edge),
        "markets": {
            "spread_home": summarize_market(best, "spread_home"),
            "total_over": summarize_market(best, "total_over"),
            "total_under": summarize_market(best, "total_under"),
        },
        "notes": [
            "Probabilities are calibrated walk-forward using prior seasons only when enough prior rows exist.",
            "Current spread schema grades the home side because a separate away-side quote is not required.",
            "Totals choose the stronger of Over/Under per game after calibration.",
            "Positive ROI in a historical sample is not proof of future profitability.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
