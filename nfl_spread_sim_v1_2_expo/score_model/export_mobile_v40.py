from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def parse_teams(game_id: str):
    parts = str(game_id).split("_")
    if len(parts) >= 4:
        return parts[-2], parts[-1]
    return "AWAY", "HOME"


def market_pick(row: pd.Series | None):
    if row is None:
        return None

    return {
        "market": str(row["market"]).lower(),
        "pick": str(row["pick"]),
        "line": clean(row.get("line")),
        "odds": clean(row.get("odds")),
        "book": str(row.get("book", "")),
        "rawProbability": clean(row.get("raw_probability")),
        "calibratedProbability": clean(row.get("calibrated_probability")),
        "selectedProbability": clean(row.get("selected_probability")),
        "breakEven": clean(row.get("break_even")),
        "edge": clean(row.get("edge")),
        "minimumEdge": clean(row.get("minimum_edge")),
        "qualifies": truthy(row.get("qualifies")),
        "quoteAgeMinutes": clean(row.get("quote_age_minutes")),
    }


def load_optional(path: str):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def first_row(df: pd.DataFrame, market: str):
    x = df[df["market"].astype(str).str.lower().eq(market)]
    if x.empty:
        return None
    return x.iloc[0]


def build_games(best: pd.DataFrame, shadow: pd.DataFrame):
    shadow_map = {}
    if not shadow.empty and "game_id" in shadow.columns:
        for _, r in shadow.iterrows():
            shadow_map[str(r["game_id"])] = r

    games = []

    for game_id, g in best.groupby("game_id"):
        spread_row = first_row(g, "spread")
        total_row = first_row(g, "total")
        exemplar = spread_row if spread_row is not None else total_row

        away, home = parse_teams(game_id)

        if exemplar is not None:
            away = str(exemplar.get("away_team", away))
            home = str(exemplar.get("home_team", home))

        spread = market_pick(spread_row)
        total = market_pick(total_row)

        edges = [
            p["edge"]
            for p in [spread, total]
            if p and p["edge"] is not None
        ]

        qualifies = any(
            p is not None and p["qualifies"]
            for p in [spread, total]
        )

        sr = shadow_map.get(str(game_id))
        shadow_total = None
        if sr is not None:
            shadow_total = {
                "currentConsensusTotal": clean(
                    sr.get("current_consensus_total")
                ),
                "shadowTotal": clean(
                    sr.get("shadow_stack_total_current_proxy")
                ),
                "gap": clean(
                    sr.get("shadow_gap_vs_current_consensus")
                ),
                "direction": str(sr.get("shadow_direction", "PASS")).upper(),
                "booksInConsensus": int(
                    clean(sr.get("books_in_consensus")) or 0
                ),
                "timingMatchedToTraining": truthy(
                    sr.get("timing_matched_to_training")
                ),
                "shadowOnly": truthy(sr.get("shadow_only", True)),
            }

        games.append({
            "id": str(game_id),
            "season": int(clean(exemplar.get("season")) or 2026),
            "week": int(clean(exemplar.get("week")) or 3),
            "away": away,
            "home": home,
            "awayQB": clean(exemplar.get("away_qb_name")),
            "homeQB": clean(exemplar.get("home_qb_name")),
            "modelMargin": clean(exemplar.get("model_margin")),
            "modelTotal": clean(exemplar.get("model_total")),
            "spread": spread,
            "total": total,
            "shadowTotal": shadow_total,
            "qualifies": qualifies,
            "bestEdge": max(edges) if edges else None,
        })

    games.sort(
        key=lambda x: (
            not x["qualifies"],
            -(x["bestEdge"] if x["bestEdge"] is not None else -999),
            x["id"],
        )
    )
    return games


def build_forward(ledger: pd.DataFrame):
    if ledger.empty:
        return [], {
            "qualifiedCount": 0,
            "settledCount": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "forwardUnits": 0.0,
            "placedCount": 0,
            "clvCount": 0,
            "positiveClvCount": 0,
            "averageClvPoints": None,
        }

    rows = []

    for _, r in ledger.iterrows():
        rows.append({
            "gameId": str(r.get("game_id")),
            "market": str(r.get("market", "")),
            "pick": str(r.get("pick", "")),
            "line": clean(r.get("line")),
            "odds": clean(r.get("odds")),
            "book": str(r.get("book", "")),
            "edge": clean(r.get("edge")),
            "placed": truthy(r.get("placed")),
            "stakeUnits": clean(r.get("stake_units")),
            "closeConsensusLine": clean(r.get("close_consensus_line")),
            "closeBestLine": clean(r.get("close_best_line")),
            "clvPointsVsConsensus": clean(r.get("clv_points_vs_consensus")),
            "clvPointsVsBest": clean(r.get("clv_points_vs_best")),
            "result": clean(r.get("result")),
            "settlementProfitUnits": clean(
                r.get("settlement_profit_units")
            ),
            "status": str(r.get("status", "")),
            "closeNote": clean(r.get("close_note")),
        })

    results = ledger.get("result", pd.Series(dtype=object)).astype(str).str.lower()
    settled_mask = results.isin(["win", "loss", "push"])

    clv = pd.to_numeric(
        ledger.get(
            "clv_points_vs_consensus",
            pd.Series(index=ledger.index, dtype=float),
        ),
        errors="coerce",
    ).dropna()

    profits = pd.to_numeric(
        ledger.get(
            "settlement_profit_units",
            pd.Series(index=ledger.index, dtype=float),
        ),
        errors="coerce",
    ).dropna()

    summary = {
        "qualifiedCount": int(len(ledger)),
        "settledCount": int(settled_mask.sum()),
        "wins": int((results == "win").sum()),
        "losses": int((results == "loss").sum()),
        "pushes": int((results == "push").sum()),
        "forwardUnits": float(profits.sum()) if len(profits) else 0.0,
        "placedCount": int(
            ledger.get(
                "placed",
                pd.Series(False, index=ledger.index),
            ).map(truthy).sum()
        ),
        "clvCount": int(len(clv)),
        "positiveClvCount": int((clv > 0).sum()),
        "averageClvPoints": float(clv.mean()) if len(clv) else None,
    }

    return rows, summary


def build_historical(metrics: pd.DataFrame):
    if metrics.empty:
        return []

    out = []
    for _, r in metrics.iterrows():
        market = str(r.get("market", "")).lower()
        if market not in {"spread", "total"}:
            continue
        out.append({
            "market": market,
            "games": int(clean(r.get("games")) or 0),
            "marketMae": clean(r.get("market_mae")),
            "marketCalibratedMae": clean(r.get("market_calibrated_mae")),
            "baseModelMae": clean(r.get("base_model_mae")),
            "stackMae": clean(r.get("stack_mae")),
            "marketRmse": clean(r.get("market_rmse")),
            "marketCalibratedRmse": clean(
                r.get("market_calibrated_rmse")
            ),
            "baseModelRmse": clean(r.get("base_model_rmse")),
            "stackRmse": clean(r.get("stack_rmse")),
        })
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--best",
        default="artifacts/live/week3_2026_best_v31.csv",
    )
    ap.add_argument(
        "--ledger",
        default="artifacts/live/forward_test_ledger_v39.csv",
    )
    ap.add_argument(
        "--shadow",
        default="artifacts/live/week3_2026_totals_shadow_v38.csv",
    )
    ap.add_argument(
        "--historical",
        default="artifacts/market_stack_v37/overall_metrics.csv",
    )
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=3)
    ap.add_argument(
        "--output",
        default="mobile/generatedData.ts",
    )

    args = ap.parse_args()

    best = pd.read_csv(args.best)
    ledger = load_optional(args.ledger)
    shadow = load_optional(args.shadow)
    historical = load_optional(args.historical)

    games = build_games(best, shadow)
    forward, performance = build_forward(ledger)

    # Prefer actual number of current production qualifiers from best CSV.
    production_qualifiers = 0
    if "qualifies" in best.columns:
        production_qualifiers = int(best["qualifies"].map(truthy).sum())

    performance["qualifiedCount"] = production_qualifiers

    snapshot = {
        "generatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "season": args.season,
        "week": args.week,
        "games": games,
        "forwardPicks": forward,
        "historicalMetrics": build_historical(historical),
        "performance": performance,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    body = (
        'import type { AppSnapshot } from "./types";\n\n'
        "export const APP_DATA = "
        + json.dumps(snapshot, indent=2, ensure_ascii=False)
        + " satisfies AppSnapshot;\n"
    )
    output.write_text(body, encoding="utf-8")

    print("\nV4.0 MOBILE DATA EXPORT")
    print(f"Games: {len(games)}")
    print(f"Production qualifiers: {production_qualifiers}")
    print(f"Forward-test rows: {len(forward)}")
    print(f"Historical metric rows: {len(snapshot['historicalMetrics'])}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
