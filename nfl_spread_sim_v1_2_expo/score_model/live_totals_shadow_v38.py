from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .historical_market_timing_v38 import audit_market


def fit_frozen_2026_stack(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
):
    m = market[
        ["game_id", "total_line", "closing_total"]
    ].drop_duplicates("game_id").copy()

    d = predictions.merge(m, on="game_id", how="inner")

    required = [
        "season",
        "actual_total",
        "pred_total",
        "total_line",
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(
            f"Historical stack training is missing: {missing}"
        )

    d = d[
        pd.to_numeric(d["season"], errors="coerce") <= 2025
    ].copy()

    d = d.dropna(
        subset=[
            "actual_total",
            "pred_total",
            "total_line",
        ]
    )

    x = d[["total_line", "pred_total"]].to_numpy(float)
    y = d["actual_total"].to_numpy(float)

    model = LinearRegression()
    model.fit(x, y)

    market_only = LinearRegression()
    market_only.fit(
        d[["total_line"]].to_numpy(float),
        y,
    )

    return {
        "rows": int(len(d)),
        "min_season": int(d["season"].min()),
        "max_season": int(d["season"].max()),
        "stack_intercept": float(model.intercept_),
        "stack_market_coef": float(model.coef_[0]),
        "stack_model_coef": float(model.coef_[1]),
        "market_only_intercept": float(market_only.intercept_),
        "market_only_coef": float(market_only.coef_[0]),
    }


def total_consensus(all_candidates: pd.DataFrame):
    q = all_candidates[
        all_candidates["market"].astype(str).str.lower().eq("total")
    ].copy()

    if "quote_fresh" in q.columns:
        q = q[q["quote_fresh"].fillna(False)].copy()

    if q.empty:
        return pd.DataFrame()

    for c in ["line", "odds", "model_total"]:
        q[c] = pd.to_numeric(q[c], errors="coerce")

    q = q.dropna(
        subset=["game_id", "book", "line", "model_total"]
    )

    # Collapse over/under duplicates to one line per book/game.
    per_book = (
        q.groupby(
            ["game_id", "away_team", "home_team", "book"],
            as_index=False,
        )
        .agg(
            book_total_line=("line", "median"),
            model_total=("model_total", "median"),
            latest_quote_at=("quote_updated_at", "max"),
        )
    )

    rows = []

    for game_id, g in per_book.groupby("game_id"):
        rows.append({
            "game_id": game_id,
            "away_team": g["away_team"].iloc[0],
            "home_team": g["home_team"].iloc[0],
            "books_in_consensus": int(g["book"].nunique()),
            "current_consensus_total": float(
                g["book_total_line"].median()
            ),
            "min_book_total": float(g["book_total_line"].min()),
            "max_book_total": float(g["book_total_line"].max()),
            "model_total": float(g["model_total"].median()),
            "latest_quote_at": g["latest_quote_at"].max(),
        })

    return pd.DataFrame(rows)


def best_directional_prices(
    all_candidates: pd.DataFrame,
):
    q = all_candidates[
        all_candidates["market"].astype(str).str.lower().eq("total")
    ].copy()

    if "quote_fresh" in q.columns:
        q = q[q["quote_fresh"].fillna(False)].copy()

    if q.empty:
        return pd.DataFrame()

    q["line"] = pd.to_numeric(q["line"], errors="coerce")
    q["odds"] = pd.to_numeric(q["odds"], errors="coerce")

    rows = []

    for game_id, g in q.groupby("game_id"):
        over = g[g["side"].astype(str).str.lower().eq("over")].copy()
        under = g[g["side"].astype(str).str.lower().eq("under")].copy()

        r = {"game_id": game_id}

        if not over.empty:
            # For over, lower line is better; then better price.
            o = over.sort_values(
                ["line", "odds"],
                ascending=[True, False],
            ).iloc[0]
            r.update({
                "best_over_line": float(o["line"]),
                "best_over_odds": float(o["odds"]),
                "best_over_book": str(o["book"]),
            })

        if not under.empty:
            # For under, higher line is better; then better price.
            u = under.sort_values(
                ["line", "odds"],
                ascending=[False, False],
            ).iloc[0]
            r.update({
                "best_under_line": float(u["line"]),
                "best_under_odds": float(u["odds"]),
                "best_under_book": str(u["book"]),
            })

        rows.append(r)

    return pd.DataFrame(rows)


def build_shadow(
    consensus: pd.DataFrame,
    prices: pd.DataFrame,
    coeffs: dict,
    timing_classification: str,
    snapshot_label: str,
):
    d = consensus.merge(prices, on="game_id", how="left")

    d["shadow_stack_total_current_proxy"] = (
        coeffs["stack_intercept"]
        + coeffs["stack_market_coef"]
        * d["current_consensus_total"]
        + coeffs["stack_model_coef"]
        * d["model_total"]
    )

    d["shadow_gap_vs_current_consensus"] = (
        d["shadow_stack_total_current_proxy"]
        - d["current_consensus_total"]
    )

    d["shadow_direction"] = np.where(
        d["shadow_gap_vs_current_consensus"] > 0,
        "OVER",
        np.where(
            d["shadow_gap_vs_current_consensus"] < 0,
            "UNDER",
            "PASS",
        ),
    )

    d["historical_market_timing"] = timing_classification
    d["live_market_timing"] = "current_multibook_consensus_proxy"
    d["timing_matched_to_training"] = (
        timing_classification != "closing_alias"
    )

    d["snapshot_label"] = snapshot_label
    d["shadow_only"] = True
    d["production_action"] = "NO_CHANGE_TO_V3_1"

    d["snapshot_created_utc"] = pd.Timestamp.now(
        tz="UTC"
    ).isoformat()

    return d


def append_log(shadow: pd.DataFrame, log_csv: str):
    p = Path(log_csv)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists():
        old = pd.read_csv(p)
        new = pd.concat([old, shadow], ignore_index=True)
    else:
        new = shadow.copy()

    new.to_csv(p, index=False)
    return p


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--historical-predictions",
        default=(
            "artifacts/dual_market/"
            "oos_qb_raw_context_v28.csv"
        ),
    )
    ap.add_argument(
        "--historical-market",
        default="data/historical_market.csv",
    )
    ap.add_argument(
        "--live-all-candidates",
        default=(
            "artifacts/live/"
            "week3_2026_all_candidates_v31.csv"
        ),
    )
    ap.add_argument(
        "--snapshot-label",
        default="current",
    )
    ap.add_argument(
        "--output",
        default=(
            "artifacts/live/"
            "week3_2026_totals_shadow_v38.csv"
        ),
    )
    ap.add_argument(
        "--log-csv",
        default=(
            "artifacts/live/"
            "totals_shadow_history_v38.csv"
        ),
    )
    ap.add_argument(
        "--coefficients-json",
        default=(
            "artifacts/live/"
            "totals_stack_coefficients_2026_v38.json"
        ),
    )

    args = ap.parse_args()

    hist_pred = pd.read_csv(args.historical_predictions)
    hist_market = pd.read_csv(args.historical_market)
    live = pd.read_csv(args.live_all_candidates)

    audit, timing = audit_market(hist_market)
    coeffs = fit_frozen_2026_stack(
        hist_pred,
        hist_market,
    )

    consensus = total_consensus(live)
    if consensus.empty:
        raise SystemExit(
            "No fresh total quotes found in the v3.1 candidate file."
        )

    prices = best_directional_prices(live)

    shadow = build_shadow(
        consensus,
        prices,
        coeffs,
        timing,
        args.snapshot_label,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    shadow.to_csv(out, index=False)

    coeff_path = Path(args.coefficients_json)
    coeff_path.parent.mkdir(parents=True, exist_ok=True)
    coeff_path.write_text(
        json.dumps(
            {
                **coeffs,
                "historical_market_timing": timing,
                "note": (
                    "These 2026 coefficients were trained using historical "
                    "market total_line. If the timing audit classifies that "
                    "field as closing_alias, current live consensus is only "
                    "a shadow proxy until closing-time validation exists."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log_path = append_log(shadow, args.log_csv)

    print("\nV3.8 HISTORICAL TIMING")
    print(audit.to_string(index=False))
    print("Classification:", timing)

    print("\nV3.8 FROZEN 2026 TOTAL STACK")
    print(json.dumps(coeffs, indent=2))

    show = [
        "game_id",
        "away_team",
        "home_team",
        "books_in_consensus",
        "current_consensus_total",
        "model_total",
        "shadow_stack_total_current_proxy",
        "shadow_gap_vs_current_consensus",
        "shadow_direction",
        "best_over_line",
        "best_over_odds",
        "best_over_book",
        "best_under_line",
        "best_under_odds",
        "best_under_book",
        "historical_market_timing",
        "timing_matched_to_training",
        "snapshot_label",
        "shadow_only",
    ]
    show = [c for c in show if c in shadow.columns]

    print("\nV3.8 TOTALS SHADOW")
    print(
        shadow[show]
        .sort_values(
            "shadow_gap_vs_current_consensus",
            key=lambda s: s.abs(),
            ascending=False,
        )
        .to_string(index=False)
    )

    print(
        "\nIMPORTANT: this is SHADOW ONLY. "
        "If historical timing is closing_alias, the current consensus "
        "input does not match training-time market information."
    )
    print(f"\nSaved current shadow: {out}")
    print(f"Appended shadow history: {log_path}")
    print(f"Saved frozen coefficients: {coeff_path}")


if __name__ == "__main__":
    main()
