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


def week_key(season: pd.Series, week: pd.Series) -> pd.Series:
    return season.astype(int) * 100 + week.astype(int)


def empirical_prob(proj: float, residuals: np.ndarray, threshold: float, gt: bool):
    sims = proj + residuals
    push = np.isclose(sims, threshold)
    win = sims > threshold if gt else sims < threshold
    return float(win.mean()), float(push.mean())


def spread_grade(actual_margin: float, home_spread: float):
    v = actual_margin + home_spread
    if np.isclose(v, 0): return "push"
    return "home" if v > 0 else "away"


def total_grade(actual_total: float, total_line: float):
    v = actual_total - total_line
    if np.isclose(v, 0): return "push"
    return "over" if v > 0 else "under"


def build_week_safe_rows(market: pd.DataFrame, pred: pd.DataFrame, min_prior_games=50):
    p = ensure_prediction_columns(pred)
    if "week" not in p.columns:
        raise ValueError("OOS predictions must include week for week-safe backtesting.")
    if "week" not in market.columns:
        raise ValueError("Market data must include week for week-safe backtesting.")

    p["week_key"] = week_key(p["season"], p["week"])
    market = market.copy()
    market["week_key"] = week_key(market["season"], market["week"])

    merged = market.merge(
        p[[
            "game_id","season","week","week_key","pred_margin","pred_total",
            "actual_margin","actual_total"
        ]],
        on=["game_id","season","week","week_key"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["week_key","game_id"]).reset_index(drop=True)

    rows = []
    for _, r in merged.iterrows():
        # CRITICAL: prior WEEKS only, never another game from current week.
        prior = merged[merged["week_key"] < r["week_key"]]
        if len(prior) < min_prior_games:
            continue

        prior_actual_margin = prior["home_score"] - prior["away_score"]
        prior_actual_total = prior["home_score"] + prior["away_score"]

        mres = (prior_actual_margin - prior["pred_margin"]).dropna().to_numpy(float)
        tres = (prior_actual_total - prior["pred_total"]).dropna().to_numpy(float)

        am = float(r["home_score"] - r["away_score"])
        at = float(r["home_score"] + r["away_score"])
        sg = spread_grade(am, float(r["home_spread"]))
        tg = total_grade(at, float(r["total_line"]))

        hp,hpp = empirical_prob(float(r["pred_margin"]),mres,-float(r["home_spread"]),True)
        ap,app = empirical_prob(float(r["pred_margin"]),mres,-float(r["home_spread"]),False)
        op,opp = empirical_prob(float(r["pred_total"]),tres,float(r["total_line"]),True)
        up,upp = empirical_prob(float(r["pred_total"]),tres,float(r["total_line"]),False)

        common = dict(
            game_id=r["game_id"], season=int(r["season"]), week=int(r["week"]),
            week_key=int(r["week_key"]), prior_residual_games=int(len(prior)),
        )
        rows.extend([
            dict(**common,market="spread",side="home",selection=str(r["home_team"]),
                 line=float(r["home_spread"]),odds=float(r["home_spread_odds"]),
                 raw_probability=hp,push_probability=hpp,
                 break_even_probability=break_even(r["home_spread_odds"]),
                 line_edge_points=float(r["pred_margin"]+r["home_spread"]),
                 won=int(sg=="home"),is_push=bool(sg=="push")),
            dict(**common,market="spread",side="away",selection=str(r["away_team"]),
                 line=float(r["away_spread"]),odds=float(r["away_spread_odds"]),
                 raw_probability=ap,push_probability=app,
                 break_even_probability=break_even(r["away_spread_odds"]),
                 line_edge_points=float(-(r["pred_margin"]+r["home_spread"])),
                 won=int(sg=="away"),is_push=bool(sg=="push")),
            dict(**common,market="total",side="over",selection="OVER",
                 line=float(r["total_line"]),odds=float(r["over_odds"]),
                 raw_probability=op,push_probability=opp,
                 break_even_probability=break_even(r["over_odds"]),
                 line_edge_points=float(r["pred_total"]-r["total_line"]),
                 won=int(tg=="over"),is_push=bool(tg=="push")),
            dict(**common,market="total",side="under",selection="UNDER",
                 line=float(r["total_line"]),odds=float(r["under_odds"]),
                 raw_probability=up,push_probability=upp,
                 break_even_probability=break_even(r["under_odds"]),
                 line_edge_points=float(r["total_line"]-r["pred_total"]),
                 won=int(tg=="under"),is_push=bool(tg=="push")),
        ])
    return pd.DataFrame(rows)


def calibrate_prior_weeks_only(rows: pd.DataFrame, min_rows=100):
    d = rows.copy().sort_values(["week_key","game_id","market","side"]).reset_index(drop=True)
    d["calibrated_probability"] = d["raw_probability"]

    for idx,r in d.iterrows():
        hist = d[
            (d["week_key"] < r["week_key"]) &
            (d["market"] == r["market"]) &
            (d["side"] == r["side"]) &
            (~d["is_push"])
        ]
        if len(hist) < min_rows or hist["won"].nunique() < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(hist["raw_probability"], hist["won"])
        d.loc[idx,"calibrated_probability"] = float(iso.predict([r["raw_probability"]])[0])

    d["probability_edge"] = d["calibrated_probability"] - d["break_even_probability"]
    d["profit_units"] = np.where(
        d["is_push"], 0.0,
        np.where(d["won"].eq(1), d["odds"].map(win_profit), -1.0)
    )
    return d


def select_best(d):
    return (
        d.sort_values(["game_id","market","probability_edge","line_edge_points"],
                      ascending=[True,True,False,False])
         .drop_duplicates(["game_id","market"])
         .reset_index(drop=True)
    )


def summarize(df):
    settled=df[~df["is_push"]]
    return {
        "bets":int(len(df)),
        "settled":int(len(settled)),
        "pushes":int(df["is_push"].sum()),
        "win_rate":float(settled["won"].mean()) if len(settled) else None,
        "units":float(df["profit_units"].sum()),
        "roi":float(settled["profit_units"].mean()) if len(settled) else None,
        "avg_edge":float(df["probability_edge"].mean()) if len(df) else None,
        "brier":float(brier_score_loss(settled["won"],settled["calibrated_probability"]))
          if len(settled) else None,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--market-csv",default="data/historical_market.csv")
    ap.add_argument("--oos-predictions",default="artifacts/dual_market/walk_forward_predictions.csv")
    ap.add_argument("--output-dir",default="artifacts/market_backtest_v16")
    ap.add_argument("--min-edge",type=float,default=0.0)
    ap.add_argument("--min-prior-games",type=int,default=50)
    ap.add_argument("--min-calibration-rows",type=int,default=100)
    args=ap.parse_args()

    market=pd.read_csv(args.market_csv)
    pred=pd.read_csv(args.oos_predictions)

    rows=build_week_safe_rows(market,pred,args.min_prior_games)
    rows=calibrate_prior_weeks_only(rows,args.min_calibration_rows)
    best=select_best(rows)
    bets=best[best["probability_edge"]>=args.min_edge].copy()

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows.to_csv(out/"all_sides.csv",index=False)
    best.to_csv(out/"best_per_game_market.csv",index=False)
    bets.to_csv(out/"bettable.csv",index=False)

    report={
        "matched_games_after_warmup":int(best.game_id.nunique()),
        "selected_bets":int(len(bets)),
        "week_safe":True,
        "same_week_results_used":False,
        "spread":summarize(bets[bets.market.eq("spread")]),
        "total":summarize(bets[bets.market.eq("total")]),
    }
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
