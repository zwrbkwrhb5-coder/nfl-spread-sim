from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def break_even(odds: float) -> float:
    odds = float(odds)
    return 100/(odds+100) if odds > 0 else (-odds)/((-odds)+100)


def win_profit(odds: float) -> float:
    odds = float(odds)
    return odds/100 if odds > 0 else 100/(-odds)


def spread_grade(actual_margin: float, home_spread: float) -> str:
    v = actual_margin + home_spread
    if np.isclose(v, 0):
        return "push"
    return "home" if v > 0 else "away"


def total_grade(actual_total: float, total_line: float) -> str:
    v = actual_total - total_line
    if np.isclose(v, 0):
        return "push"
    return "over" if v > 0 else "under"


def empirical_prob(proj: float, residuals: np.ndarray, threshold: float, gt: bool):
    if len(residuals) == 0:
        return np.nan, np.nan
    sims = proj + residuals
    pushes = np.isclose(sims, threshold)
    wins = sims > threshold if gt else sims < threshold
    return float(wins.mean()), float(pushes.mean())


def ensure_prediction_columns(pred: pd.DataFrame) -> pd.DataFrame:
    p = pred.copy()
    if "pred_margin" not in p:
        p["pred_margin"] = p["pred_home_score"] - p["pred_away_score"]
    if "pred_total" not in p:
        p["pred_total"] = p["pred_home_score"] + p["pred_away_score"]

    if "actual_home_score" not in p and "home_score" in p:
        p["actual_home_score"] = p["home_score"]
    if "actual_away_score" not in p and "away_score" in p:
        p["actual_away_score"] = p["away_score"]

    if "actual_margin" not in p:
        p["actual_margin"] = p["actual_home_score"] - p["actual_away_score"]
    if "actual_total" not in p:
        p["actual_total"] = p["actual_home_score"] + p["actual_away_score"]
    return p


def add_order_key(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "week" in d:
        d["order_key"] = d["season"].astype(int) * 100 + d["week"].fillna(0).astype(int)
    else:
        d["order_key"] = d["season"].astype(int) * 100
    return d


def build_leakage_safe_rows(
    market: pd.DataFrame,
    pred: pd.DataFrame,
    min_residual_games: int = 50,
) -> pd.DataFrame:
    p = ensure_prediction_columns(pred)
    p = add_order_key(p)
    m = add_order_key(market)

    join_cols = ["game_id", "season"]
    keep = [
        "game_id","season","pred_margin","pred_total",
        "actual_margin","actual_total","order_key"
    ]
    if "week" in p.columns:
        keep.append("week")

    merged = m.merge(
        p[keep],
        on=join_cols,
        how="inner",
        suffixes=("","_pred"),
        validate="one_to_one",
    ).sort_values(["season","week" if "week" in m.columns else "game_id","game_id"]).reset_index(drop=True)

    rows = []
    for i, r in merged.iterrows():
        # STRICTLY prior games only.
        prior = merged.iloc[:i]
        prior_margin_resid = (prior["actual_margin"] - prior["pred_margin"]).dropna().to_numpy(float)
        prior_total_resid = (prior["actual_total"] - prior["pred_total"]).dropna().to_numpy(float)

        if len(prior_margin_resid) < min_residual_games or len(prior_total_resid) < min_residual_games:
            continue

        am = float(r["home_score"] - r["away_score"])
        at = float(r["home_score"] + r["away_score"])
        sg = spread_grade(am, float(r["home_spread"]))
        tg = total_grade(at, float(r["total_line"]))

        hp, hpp = empirical_prob(float(r["pred_margin"]), prior_margin_resid, -float(r["home_spread"]), True)
        ap, app = empirical_prob(float(r["pred_margin"]), prior_margin_resid, -float(r["home_spread"]), False)
        op, opp = empirical_prob(float(r["pred_total"]), prior_total_resid, float(r["total_line"]), True)
        up, upp = empirical_prob(float(r["pred_total"]), prior_total_resid, float(r["total_line"]), False)

        common = dict(
            game_id=r["game_id"], season=int(r["season"]),
            week=int(r["week"]) if "week" in r and pd.notna(r["week"]) else None,
            prior_residual_games=int(len(prior)),
        )

        rows += [
            dict(**common, market="spread", selection=str(r["home_team"]), side="home",
                 line=float(r["home_spread"]), odds=float(r["home_spread_odds"]),
                 raw_probability=hp, push_probability=hpp,
                 break_even_probability=break_even(r["home_spread_odds"]),
                 line_edge_points=float(r["pred_margin"] + r["home_spread"]),
                 won=int(sg=="home"), is_push=bool(sg=="push")),
            dict(**common, market="spread", selection=str(r["away_team"]), side="away",
                 line=float(r["away_spread"]), odds=float(r["away_spread_odds"]),
                 raw_probability=ap, push_probability=app,
                 break_even_probability=break_even(r["away_spread_odds"]),
                 line_edge_points=float(-(r["pred_margin"] + r["home_spread"])),
                 won=int(sg=="away"), is_push=bool(sg=="push")),
            dict(**common, market="total", selection="OVER", side="over",
                 line=float(r["total_line"]), odds=float(r["over_odds"]),
                 raw_probability=op, push_probability=opp,
                 break_even_probability=break_even(r["over_odds"]),
                 line_edge_points=float(r["pred_total"] - r["total_line"]),
                 won=int(tg=="over"), is_push=bool(tg=="push")),
            dict(**common, market="total", selection="UNDER", side="under",
                 line=float(r["total_line"]), odds=float(r["under_odds"]),
                 raw_probability=up, push_probability=upp,
                 break_even_probability=break_even(r["under_odds"]),
                 line_edge_points=float(r["total_line"] - r["pred_total"]),
                 won=int(tg=="under"), is_push=bool(tg=="push")),
        ]

    return pd.DataFrame(rows)


def calibrate_strict_walk_forward(rows: pd.DataFrame, min_cal_rows: int = 100) -> pd.DataFrame:
    d = rows.copy().sort_values(["season","week","game_id"]).reset_index(drop=True)
    d["calibrated_probability"] = d["raw_probability"]

    for idx, r in d.iterrows():
        hist = d.iloc[:idx]
        hist = hist[(hist["market"] == r["market"]) & (hist["side"] == r["side"]) & (~hist["is_push"])]
        if len(hist) < min_cal_rows or hist["won"].nunique() < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(hist["raw_probability"], hist["won"])
        d.loc[idx, "calibrated_probability"] = float(iso.predict([r["raw_probability"]])[0])

    d["probability_edge"] = d["calibrated_probability"] - d["break_even_probability"]
    d["profit_units"] = np.where(
        d["is_push"], 0.0,
        np.where(d["won"].eq(1), d["odds"].map(win_profit), -1.0)
    )
    return d


def select_best(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.sort_values(
            ["game_id","market","probability_edge","line_edge_points"],
            ascending=[True,True,False,False]
        )
        .drop_duplicates(["game_id","market"])
        .reset_index(drop=True)
    )


def wilson_interval(wins: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    phat = wins/n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n))/denom
    margin = z*np.sqrt((phat*(1-phat) + z*z/(4*n))/n)/denom
    return float(center-margin), float(center+margin)


def bootstrap_roi_ci(profits: np.ndarray, n_boot: int = 2000, seed: int = 7):
    if len(profits) == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(profits, size=len(profits), replace=True)
        vals.append(sample.mean())
    return float(np.quantile(vals,.025)), float(np.quantile(vals,.975))


def summarize(df: pd.DataFrame) -> dict:
    settled = df[~df["is_push"]].copy()
    wins = int(settled["won"].sum())
    n = int(len(settled))
    lo, hi = wilson_interval(wins, n)
    roi_lo, roi_hi = bootstrap_roi_ci(settled["profit_units"].to_numpy(float))
    return {
        "bets": int(len(df)),
        "settled": n,
        "pushes": int(df["is_push"].sum()),
        "win_rate": float(settled["won"].mean()) if n else None,
        "win_rate_95ci": [lo, hi],
        "units": float(df["profit_units"].sum()),
        "roi": float(settled["profit_units"].mean()) if n else None,
        "roi_95ci_bootstrap": [roi_lo, roi_hi],
        "avg_edge": float(df["probability_edge"].mean()) if len(df) else None,
        "brier": float(brier_score_loss(settled["won"], settled["calibrated_probability"])) if n else None,
    }


def grouped_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        s = summarize(g)
        row = dict(zip(group_cols, keys))
        row.update(s)
        rows.append(row)
    return pd.DataFrame(rows)


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["edge_bucket"] = pd.cut(
        d["probability_edge"],
        [-np.inf,0,.01,.02,.03,.05,.075,.10,np.inf],
        labels=["<=0%","0-1%","1-2%","2-3%","3-5%","5-7.5%","7.5-10%","10%+"],
        right=False
    )
    # Spread-specific buckets
    d["line_size_bucket"] = pd.cut(
        d["line"].abs(),
        [-np.inf,3,7,10,np.inf],
        labels=["<3","3-6.5","7-9.5","10+"],
        right=False
    )
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-csv", default="data/historical_market.csv")
    ap.add_argument("--oos-predictions", default="artifacts/dual_market/oos_predictions.csv")
    ap.add_argument("--output-dir", default="artifacts/market_backtest_v15")
    ap.add_argument("--min-edge", type=float, default=0.0)
    ap.add_argument("--min-residual-games", type=int, default=50)
    ap.add_argument("--min-calibration-rows", type=int, default=100)
    args = ap.parse_args()

    market = pd.read_csv(args.market_csv)
    pred = pd.read_csv(args.oos_predictions)

    rows = build_leakage_safe_rows(
        market, pred, min_residual_games=args.min_residual_games
    )
    rows = calibrate_strict_walk_forward(
        rows, min_cal_rows=args.min_calibration_rows
    )
    rows = add_buckets(rows)
    best = select_best(rows)
    bettable = best[best["probability_edge"] >= args.min_edge].copy()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows.to_csv(out/"all_sides.csv", index=False)
    best.to_csv(out/"best_per_game_market.csv", index=False)
    bettable.to_csv(out/"bettable.csv", index=False)

    grouped_summary(bettable, ["market"]).to_csv(out/"summary_by_market.csv", index=False)
    grouped_summary(bettable, ["market","season"]).to_csv(out/"summary_by_season.csv", index=False)
    grouped_summary(bettable, ["market","edge_bucket"]).to_csv(out/"summary_by_edge_bucket.csv", index=False)

    spread = bettable[bettable["market"].eq("spread")].copy()
    if not spread.empty:
        grouped_summary(spread, ["side"]).to_csv(out/"spread_home_away.csv", index=False)
        grouped_summary(spread, ["line_size_bucket"]).to_csv(out/"spread_by_line_size.csv", index=False)

    report = {
        "matched_games_after_warmup": int(best["game_id"].nunique()) if not best.empty else 0,
        "selected_bets": int(len(bettable)),
        "min_edge": float(args.min_edge),
        "strict_walk_forward": True,
        "residuals_use_future_games": False,
        "calibration_uses_future_games": False,
        "spread": summarize(bettable[bettable["market"].eq("spread")]),
        "total": summarize(bettable[bettable["market"].eq("total")]),
        "notes": [
            "Each game uses only residuals from earlier games in chronological order.",
            "Probability calibration uses only earlier rows for the same market/side.",
            "ROI confidence intervals are bootstrap intervals, not guarantees of future profitability.",
        ],
    }
    (out/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
