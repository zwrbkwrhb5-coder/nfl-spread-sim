from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KICKOFFS = {
    "2026_03_KC_MIA": "2026-09-27T13:00:00-04:00",
    "2026_03_CAR_CLE": "2026-09-27T13:00:00-04:00",
    "2026_03_TEN_NYG": "2026-09-27T13:00:00-04:00",
    "2026_03_NE_JAX": "2026-09-27T13:00:00-04:00",
    "2026_03_LAC_BUF": "2026-09-27T13:00:00-04:00",
    "2026_03_NYJ_DET": "2026-09-27T13:00:00-04:00",
    "2026_03_HOU_IND": "2026-09-27T13:00:00-04:00",
    "2026_03_SEA_WAS": "2026-09-27T13:00:00-04:00",
    "2026_03_CIN_PIT": "2026-09-27T13:00:00-04:00",
    "2026_03_ARI_SF": "2026-09-27T16:05:00-04:00",
    "2026_03_MIN_TB": "2026-09-27T16:05:00-04:00",
    "2026_03_LV_NO": "2026-09-27T16:25:00-04:00",
    "2026_03_BAL_DAL": "2026-09-27T16:25:00-04:00",
    "2026_03_LA_DEN": "2026-09-27T20:20:00-04:00",
    "2026_03_PHI_CHI": "2026-09-28T20:15:00-04:00",
}


def clean(v: Any):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if pd.isna(v):
        return None
    return v


def truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def optional_csv(path: str):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def parse_teams(game_id: str):
    p = str(game_id).split("_")
    return (p[-2], p[-1]) if len(p) >= 4 else ("AWAY", "HOME")


def same_num(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def normalize_market(v: Any) -> str:
    s = str(v or "").strip().lower().replace("_", "").replace("-", "")
    if s in {"ml", "moneyline", "moneylines"}:
        return "moneyline"
    if s in {"spread", "spreads"}:
        return "spread"
    if s in {"total", "totals", "ou", "overunder"}:
        return "total"
    return s


def normalize_line(market: str, value: Any):
    x = clean(value)
    if market == "moneyline" and x is None:
        return 0.0
    return x


def build_odds_board(all_candidates, best_row):
    if best_row is None or all_candidates.empty:
        return []

    gid = str(best_row.get("game_id"))
    market = normalize_market(best_row.get("market"))
    pick = str(best_row.get("pick", ""))

    x = all_candidates.copy()
    x["_market_norm"] = x["market"].map(normalize_market)
    x = x[
        x["game_id"].astype(str).eq(gid)
        & x["_market_norm"].eq(market)
        & x["pick"].astype(str).eq(pick)
    ].copy()

    if x.empty:
        return []

    if "edge" in x.columns:
        x["_edge_sort"] = pd.to_numeric(x["edge"], errors="coerce").fillna(-999.0)
        x = x.sort_values("_edge_sort", ascending=False)
    x = x.drop_duplicates(subset=["book", "line", "odds"], keep="first")

    options = []
    best_line = normalize_line(market, best_row.get("line"))
    for _, r in x.iterrows():
        selected = (
            str(r.get("book", "")) == str(best_row.get("book", ""))
            and same_num(normalize_line(market, r.get("line")), best_line)
            and same_num(r.get("odds"), best_row.get("odds"))
        )
        options.append({
            "line": normalize_line(market, r.get("line")),
            "odds": clean(r.get("odds")),
            "book": str(r.get("book", "")),
            "breakEven": clean(r.get("break_even")),
            "edge": clean(r.get("edge")),
            "quoteAgeMinutes": clean(r.get("quote_age_minutes")),
            "selected": selected,
        })

    options.sort(
        key=lambda q: (
            not q["selected"],
            -(q["edge"] if q["edge"] is not None else -999.0),
            q["book"],
        )
    )

    return options[:8]


def market_pick(r, all_candidates):
    if r is None:
        return None
    market = normalize_market(r.get("market"))
    return {
        "market": market,
        "pick": str(r["pick"]),
        "line": normalize_line(market, r.get("line")),
        "odds": clean(r.get("odds")),
        "book": str(r.get("book", "")),
        "rawProbability": clean(r.get("raw_probability")),
        "calibratedProbability": clean(r.get("calibrated_probability")),
        "selectedProbability": clean(r.get("selected_probability")),
        "breakEven": clean(r.get("break_even")),
        "edge": clean(r.get("edge")),
        "minimumEdge": clean(r.get("minimum_edge")),
        "qualifies": truthy(r.get("qualifies")),
        "quoteAgeMinutes": clean(r.get("quote_age_minutes")),
        "oddsBoard": build_odds_board(all_candidates, r),
    }


def one_market(df, market):
    x = df[df["market"].map(normalize_market).eq(market)]
    return None if x.empty else x.iloc[0]


def build_games(best, all_candidates, shadow, season, week):
    shadow_map = {}
    if not shadow.empty and "game_id" in shadow.columns:
        shadow_map = {
            str(r["game_id"]): r
            for _, r in shadow.iterrows()
        }

    games = []
    for game_id, g in best.groupby("game_id"):
        spread_r = one_market(g, "spread")
        total_r = one_market(g, "total")
        moneyline_r = one_market(g, "moneyline")
        ex = spread_r if spread_r is not None else total_r if total_r is not None else moneyline_r
        away, home = parse_teams(game_id)

        spread = market_pick(spread_r, all_candidates)
        total = market_pick(total_r, all_candidates)
        moneyline = market_pick(moneyline_r, all_candidates)
        ps = [p for p in (spread, total, moneyline) if p]
        edges = [p["edge"] for p in ps if p["edge"] is not None]

        sr = shadow_map.get(str(game_id))
        sh = None
        if sr is not None:
            sh = {
                "currentConsensusTotal": clean(sr.get("current_consensus_total")),
                "shadowTotal": clean(sr.get("shadow_stack_total_current_proxy")),
                "gap": clean(sr.get("shadow_gap_vs_current_consensus")),
                "direction": str(sr.get("shadow_direction", "PASS")).upper(),
                "booksInConsensus": int(clean(sr.get("books_in_consensus")) or 0),
                "timingMatchedToTraining": truthy(sr.get("timing_matched_to_training")),
                "shadowOnly": True,
            }

        games.append({
            "id": str(game_id),
            "season": season,
            "week": week,
            "away": away,
            "home": home,
            "kickoffAt": KICKOFFS.get(str(game_id)),
            "awayQB": clean(ex.get("away_qb_name")) if ex is not None else None,
            "homeQB": clean(ex.get("home_qb_name")) if ex is not None else None,
            "modelMargin": clean(ex.get("model_margin")) if ex is not None else None,
            "modelTotal": clean(ex.get("model_total")) if ex is not None else None,
            "spread": spread,
            "total": total,
            "moneyline": moneyline,
            "shadowTotal": sh,
            "qualifies": any(p["qualifies"] for p in ps),
            "bestEdge": max(edges) if edges else None,
        })

    games.sort(key=lambda x: (x["kickoffAt"] or "9999", x["id"]))
    return games


def build_forward(ledger, results):
    result_map = {}
    if not results.empty and "game_id" in results.columns:
        result_map = {
            str(r["game_id"]): r
            for _, r in results.iterrows()
        }

    rows = []
    for _, r in ledger.iterrows():
        gid = str(r.get("game_id"))
        rr = result_map.get(gid)
        actual_margin = clean(rr.get("actual_margin")) if rr is not None else None
        actual_total = clean(rr.get("actual_total")) if rr is not None else None

        away_score = None
        home_score = None
        if actual_margin is not None and actual_total is not None:
            home_score = (actual_total + actual_margin) / 2
            away_score = (actual_total - actual_margin) / 2

        rows.append({
            "gameId": gid,
            "market": normalize_market(r.get("market", "")),
            "pick": str(r.get("pick", "")),
            "line": normalize_line(normalize_market(r.get("market", "")), r.get("line")),
            "odds": clean(r.get("odds")),
            "book": str(r.get("book", "")),
            "edge": clean(r.get("edge")),
            "selectedProbability": clean(r.get("selected_probability")),
            "breakEven": clean(r.get("break_even")),
            "placed": truthy(r.get("placed")),
            "stakeUnits": clean(r.get("stake_units")),
            "closeConsensusLine": clean(r.get("close_consensus_line")),
            "closeBestLine": clean(r.get("close_best_line")),
            "clvPointsVsConsensus": clean(r.get("clv_points_vs_consensus")),
            "clvPointsVsBest": clean(r.get("clv_points_vs_best")),
            "result": clean(r.get("result")),
            "settlementProfitUnits": clean(r.get("settlement_profit_units")),
            "status": str(r.get("status", "")),
            "closeNote": clean(r.get("close_note")),
            "actualMargin": actual_margin,
            "actualTotal": actual_total,
            "awayScore": away_score,
            "homeScore": home_score,
        })
    return rows


def performance(ledger):
    if ledger.empty:
        return {
            "qualifiedCount": 0, "settledCount": 0,
            "wins": 0, "losses": 0, "pushes": 0,
            "forwardUnits": 0, "placedCount": 0,
            "clvCount": 0, "positiveClvCount": 0,
            "averageClvPoints": None,
        }

    rs = ledger.get("result", pd.Series(dtype=object)).astype(str).str.lower()
    clv = pd.to_numeric(
        ledger.get("clv_points_vs_consensus", pd.Series(index=ledger.index, dtype=float)),
        errors="coerce",
    ).dropna()
    profits = pd.to_numeric(
        ledger.get("settlement_profit_units", pd.Series(index=ledger.index, dtype=float)),
        errors="coerce",
    ).dropna()

    return {
        "qualifiedCount": int(len(ledger)),
        "settledCount": int(rs.isin(["win", "loss", "push"]).sum()),
        "wins": int((rs == "win").sum()),
        "losses": int((rs == "loss").sum()),
        "pushes": int((rs == "push").sum()),
        "forwardUnits": float(profits.sum()) if len(profits) else 0.0,
        "placedCount": int(ledger.get("placed", pd.Series(False, index=ledger.index)).map(truthy).sum()),
        "clvCount": int(len(clv)),
        "positiveClvCount": int((clv > 0).sum()),
        "averageClvPoints": float(clv.mean()) if len(clv) else None,
    }


def historical(df):
    out = []
    if df.empty:
        return out
    for _, r in df.iterrows():
        market = normalize_market(r.get("market", ""))
        if market not in {"spread", "total", "moneyline"}:
            market = "other"
        out.append({
            "market": market,
            "games": int(clean(r.get("games")) or 0),
            "marketMae": clean(r.get("market_mae")),
            "marketCalibratedMae": clean(r.get("market_calibrated_mae")),
            "baseModelMae": clean(r.get("base_model_mae")),
            "stackMae": clean(r.get("stack_mae")),
            "marketRmse": clean(r.get("market_rmse")),
            "marketCalibratedRmse": clean(r.get("market_calibrated_rmse")),
            "baseModelRmse": clean(r.get("base_model_rmse")),
            "stackRmse": clean(r.get("stack_rmse")),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="artifacts/live/week3_2026_best_v31.csv")
    ap.add_argument("--all-candidates", default="artifacts/live/week3_2026_all_candidates_v31.csv")
    ap.add_argument("--ledger", default="artifacts/live/forward_test_ledger_v39.csv")
    ap.add_argument("--results", default="artifacts/live/week3_2026_results.csv")
    ap.add_argument("--shadow", default="artifacts/live/week3_2026_totals_shadow_v38.csv")
    ap.add_argument("--historical", default="artifacts/market_stack_v37/overall_metrics.csv")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=3)
    ap.add_argument("--output", default="mobile/generatedData.ts")
    args = ap.parse_args()

    best = pd.read_csv(args.best)
    all_candidates = optional_csv(args.all_candidates)
    ledger = optional_csv(args.ledger)
    results = optional_csv(args.results)
    shadow = optional_csv(args.shadow)
    hist = optional_csv(args.historical)

    data = {
        "generatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "season": args.season,
        "week": args.week,
        "games": build_games(best, all_candidates, shadow, args.season, args.week),
        "forwardPicks": build_forward(ledger, results),
        "historicalMetrics": historical(hist),
        "performance": performance(ledger),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        'import type { AppSnapshot } from "./types";

'
        + "export const APP_DATA = "
        + json.dumps(data, indent=2)
        + " satisfies AppSnapshot;
",
        encoding="utf-8",
    )

    board_quotes = sum(
        len(p["oddsBoard"])
        for g in data["games"]
        for p in (g["spread"], g["total"], g["moneyline"])
        if p
    )

    print("V4.9 MONEYLINE MOBILE EXPORT")
    print(f"Games: {len(data['games'])}")
    print(f"Forward picks: {len(data['forwardPicks'])}")
    print(f"Odds-board quotes: {board_quotes}")
    print(f"Record: {data['performance']['wins']}-{data['performance']['losses']}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
