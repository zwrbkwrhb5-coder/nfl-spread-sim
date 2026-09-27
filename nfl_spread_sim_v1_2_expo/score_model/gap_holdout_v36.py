from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .market_anchor_v34 import prepare


THRESHOLD = 5.0


def american_profit(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    return 100.0 / (-odds)


def wilson_interval(wins: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return np.nan, np.nan

    phat = wins / n
    denom = 1.0 + (z * z / n)
    center = (
        phat + z * z / (2.0 * n)
    ) / denom
    half = (
        z
        * np.sqrt(
            phat * (1.0 - phat) / n
            + z * z / (4.0 * n * n)
        )
        / denom
    )

    return float(center - half), float(center + half)


def add_signal_results(
    d: pd.DataFrame,
    market_raw: pd.DataFrame,
):
    mcols = [
        c for c in [
            "game_id",
            "home_spread_odds",
            "away_spread_odds",
            "over_odds",
            "under_odds",
        ]
        if c in market_raw.columns
    ]

    x = d.merge(
        market_raw[mcols].drop_duplicates("game_id"),
        on="game_id",
        how="left",
    )

    rows = []

    for _, r in x.iterrows():
        # Spread control signal
        sg = r["spread_model_gap"]
        if pd.notna(sg) and abs(float(sg)) >= THRESHOLD:
            if sg > 0:
                side = "home"
                margin_against = (
                    float(r["actual_margin"])
                    + float(r["home_spread"])
                )
                odds = r.get("home_spread_odds", np.nan)
            else:
                side = "away"
                margin_against = -(
                    float(r["actual_margin"])
                    + float(r["home_spread"])
                )
                odds = r.get("away_spread_odds", np.nan)

            result = (
                "win"
                if margin_against > 0
                else "loss"
                if margin_against < 0
                else "push"
            )

            profit = np.nan
            if pd.notna(odds):
                profit = (
                    american_profit(float(odds))
                    if result == "win"
                    else -1.0
                    if result == "loss"
                    else 0.0
                )

            rows.append({
                "game_id": r["game_id"],
                "season": int(r["season"]),
                "week": int(r["week"]),
                "week_key": int(r["week_key"]),
                "market": "spread",
                "gap": float(sg),
                "abs_gap": abs(float(sg)),
                "signal_side": side,
                "actual_market_residual": float(
                    r["spread_actual_market_residual"]
                ),
                "same_direction": bool(
                    np.sign(sg)
                    == np.sign(
                        r["spread_actual_market_residual"]
                    )
                ),
                "result": result,
                "odds": odds,
                "profit": profit,
            })

        # Total confirmatory signal
        tg = r["total_model_gap"]
        if pd.notna(tg) and abs(float(tg)) >= THRESHOLD:
            if tg > 0:
                side = "over"
                residual = (
                    float(r["actual_total"])
                    - float(r["total_line"])
                )
                odds = r.get("over_odds", np.nan)
            else:
                side = "under"
                residual = -(
                    float(r["actual_total"])
                    - float(r["total_line"])
                )
                odds = r.get("under_odds", np.nan)

            result = (
                "win"
                if residual > 0
                else "loss"
                if residual < 0
                else "push"
            )

            profit = np.nan
            if pd.notna(odds):
                profit = (
                    american_profit(float(odds))
                    if result == "win"
                    else -1.0
                    if result == "loss"
                    else 0.0
                )

            rows.append({
                "game_id": r["game_id"],
                "season": int(r["season"]),
                "week": int(r["week"]),
                "week_key": int(r["week_key"]),
                "market": "total",
                "gap": float(tg),
                "abs_gap": abs(float(tg)),
                "signal_side": side,
                "actual_market_residual": (
                    float(r["total_actual_market_residual"])
                ),
                "same_direction": bool(
                    np.sign(tg)
                    == np.sign(
                        r["total_actual_market_residual"]
                    )
                ),
                "result": result,
                "odds": odds,
                "profit": profit,
            })

    return pd.DataFrame(rows)


def summarize(signals: pd.DataFrame):
    rows = []

    for (sample, market), g in signals.groupby(
        ["sample", "market"]
    ):
        settled = g[g["result"].isin(["win", "loss"])].copy()
        wins = int((settled["result"] == "win").sum())
        n = len(settled)
        lo, hi = wilson_interval(wins, n)

        with_odds = settled[settled["profit"].notna()].copy()
        units = (
            float(with_odds["profit"].sum())
            if len(with_odds)
            else np.nan
        )
        roi = (
            units / len(with_odds)
            if len(with_odds)
            else np.nan
        )

        signed = (
            np.sign(g["gap"])
            * g["actual_market_residual"]
        )

        rows.append({
            "sample": sample,
            "market": market,
            "threshold": THRESHOLD,
            "games": len(g),
            "settled": n,
            "pushes": int((g["result"] == "push").sum()),
            "wins": wins,
            "win_rate": wins / n if n else np.nan,
            "win_rate_ci_low_95": lo,
            "win_rate_ci_high_95": hi,
            "same_direction_rate": float(
                g["same_direction"].mean()
            ),
            "avg_signed_actual_market_residual": float(
                signed.mean()
            ),
            "median_signed_actual_market_residual": float(
                signed.median()
            ),
            "priced_bets": len(with_odds),
            "units": units,
            "roi": roi,
        })

    return pd.DataFrame(rows).sort_values(
        ["market", "sample"]
    )


def week_block_bootstrap(
    signals: pd.DataFrame,
    reps=5000,
    seed=36,
):
    rng = np.random.default_rng(seed)
    rows = []

    for (sample, market), g0 in signals.groupby(
        ["sample", "market"]
    ):
        g = g0[
            g0["result"].isin(["win", "loss"])
        ].copy()

        g = g[g["profit"].notna()].copy()

        if g.empty:
            continue

        blocks = [
            (
                wk,
                float(x["profit"].sum()),
                int(len(x)),
            )
            for wk, x in g.groupby("week_key")
        ]

        if len(blocks) < 2:
            continue

        sims = []
        n_blocks = len(blocks)

        for _ in range(reps):
            idx = rng.integers(
                0,
                n_blocks,
                size=n_blocks,
            )
            units = sum(blocks[i][1] for i in idx)
            bets = sum(blocks[i][2] for i in idx)
            if bets:
                sims.append(units / bets)

        point = float(g["profit"].sum() / len(g))

        rows.append({
            "sample": sample,
            "market": market,
            "threshold": THRESHOLD,
            "settled_bets": len(g),
            "week_blocks": n_blocks,
            "roi_point_estimate": point,
            "roi_ci_low_95": float(
                np.quantile(sims, 0.025)
            ),
            "roi_ci_high_95": float(
                np.quantile(sims, 0.975)
            ),
            "bootstrap_probability_roi_positive": float(
                np.mean(np.asarray(sims) > 0)
            ),
        })

    return pd.DataFrame(rows).sort_values(
        ["market", "sample"]
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--predictions",
        default=(
            "artifacts/dual_market/"
            "oos_qb_raw_context_v28.csv"
        ),
    )
    ap.add_argument(
        "--market-csv",
        default="data/historical_market.csv",
    )
    ap.add_argument(
        "--holdout-season",
        type=int,
        default=2022,
    )
    ap.add_argument(
        "--discovery-start",
        type=int,
        default=2023,
    )
    ap.add_argument(
        "--discovery-end",
        type=int,
        default=2025,
    )
    ap.add_argument(
        "--output-dir",
        default="artifacts/gap_holdout_v36",
    )
    ap.add_argument(
        "--bootstrap-reps",
        type=int,
        default=5000,
    )

    args = ap.parse_args()

    pred = pd.read_csv(args.predictions)
    market = pd.read_csv(args.market_csv)

    prepared = prepare(pred, market)
    signals = add_signal_results(prepared, market)

    holdout = signals[
        signals["season"] == args.holdout_season
    ].copy()
    holdout["sample"] = (
        f"{args.holdout_season}_reverse_time_holdout"
    )

    discovery = signals[
        signals["season"].between(
            args.discovery_start,
            args.discovery_end,
        )
    ].copy()
    discovery["sample"] = (
        f"{args.discovery_start}_{args.discovery_end}_discovery"
    )

    combined = pd.concat(
        [holdout, discovery],
        ignore_index=True,
    )

    summary = summarize(combined)
    boot = week_block_bootstrap(
        combined,
        reps=args.bootstrap_reps,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    combined.to_csv(
        out / "gap5_signal_rows.csv",
        index=False,
    )
    summary.to_csv(
        out / "summary.csv",
        index=False,
    )
    boot.to_csv(
        out / "week_block_bootstrap_roi.csv",
        index=False,
    )

    print("\nV3.6 FIXED 5-POINT GAP TEST")
    print(
        "Threshold is frozen at abs(model - market) >= 5.0."
    )

    print("\nSUMMARY")
    print(summary.to_string(index=False))

    print("\nWEEK-BLOCK ROI BOOTSTRAP")
    print(boot.to_string(index=False))

    print(
        "\nIMPORTANT: 2022 is a reverse-time holdout for this specific "
        "5+ gap pattern, not a true future forward test. "
        "A surviving result should become a 2026 shadow signal, "
        "not an automatic production betting rule."
    )

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
