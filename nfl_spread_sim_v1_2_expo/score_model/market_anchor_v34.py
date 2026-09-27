from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def learn_weight(x, y):
    """
    Fit y ~= weight * x with no intercept.

    x = base model disagreement with market
    y = actual outcome disagreement with market

    0 means trust market completely.
    1 means trust model completely.

    Clip to [0, 1] for interpretability and to prevent extrapolating
    beyond either forecast.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.nan

    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return 0.0

    w = float(np.dot(x, y) / denom)
    return float(np.clip(w, 0.0, 1.0))


def mae(y, pred):
    return float(
        np.mean(
            np.abs(
                np.asarray(y, dtype=float)
                - np.asarray(pred, dtype=float)
            )
        )
    )


def rmse(y, pred):
    e = (
        np.asarray(y, dtype=float)
        - np.asarray(pred, dtype=float)
    )
    return float(np.sqrt(np.mean(e ** 2)))


def prepare(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
):
    p = predictions.copy()
    m = market.copy()

    # Avoid duplicate outcome columns from market.
    m = m.drop(
        columns=[
            "actual_margin",
            "actual_total",
        ],
        errors="ignore",
    )

    needed_market = [
        "game_id",
        "home_spread",
        "total_line",
    ]
    missing = [c for c in needed_market if c not in m.columns]
    if missing:
        raise ValueError(
            f"Historical market missing columns: {missing}"
        )

    keep_market = [
        c for c in m.columns
        if c in {
            "game_id",
            "home_spread",
            "total_line",
        }
    ]

    d = p.merge(
        m[keep_market].drop_duplicates("game_id"),
        on="game_id",
        how="inner",
    )

    required = [
        "game_id",
        "season",
        "week",
        "actual_margin",
        "actual_total",
        "pred_margin",
        "pred_total",
        "home_spread",
        "total_line",
    ]

    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(
            f"Merged predictions missing columns: {missing}"
        )

    if "week_key" not in d.columns:
        d["week_key"] = (
            pd.to_numeric(d["season"], errors="raise") * 100
            + pd.to_numeric(d["week"], errors="raise")
        )

    numeric = [
        "actual_margin",
        "actual_total",
        "pred_margin",
        "pred_total",
        "home_spread",
        "total_line",
        "week_key",
    ]
    for c in numeric:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Spread sign convention:
    # home -3 => sportsbook implied home margin +3.
    d["market_margin"] = -d["home_spread"]

    d["spread_model_gap"] = (
        d["pred_margin"] - d["market_margin"]
    )
    d["spread_actual_market_residual"] = (
        d["actual_margin"] - d["market_margin"]
    )

    d["total_model_gap"] = (
        d["pred_total"] - d["total_line"]
    )
    d["total_actual_market_residual"] = (
        d["actual_total"] - d["total_line"]
    )

    return d.sort_values(
        ["week_key", "game_id"]
    ).reset_index(drop=True)


def walk_forward_anchor(
    d: pd.DataFrame,
    min_prior_rows=200,
    first_eval_season=2023,
):
    out = d.copy()

    for market in ["spread", "total"]:
        out[f"{market}_anchor_weight"] = np.nan
        out[f"{market}_anchor_prior_rows"] = 0
        out[f"{market}_anchor_max_source_week"] = np.nan

    out["anchored_margin"] = np.nan
    out["anchored_total"] = np.nan

    weeks = sorted(
        out.loc[
            out["season"] >= first_eval_season,
            "week_key",
        ]
        .dropna()
        .unique()
    )

    for wk in weeks:
        prior = out[
            out["week_key"] < wk
        ].copy()

        cur_mask = out["week_key"].eq(wk)

        spread_prior = prior.dropna(
            subset=[
                "spread_model_gap",
                "spread_actual_market_residual",
            ]
        )

        if len(spread_prior) >= min_prior_rows:
            w = learn_weight(
                spread_prior["spread_model_gap"],
                spread_prior["spread_actual_market_residual"],
            )

            out.loc[
                cur_mask,
                "spread_anchor_weight",
            ] = w

            out.loc[
                cur_mask,
                "spread_anchor_prior_rows",
            ] = len(spread_prior)

            out.loc[
                cur_mask,
                "spread_anchor_max_source_week",
            ] = float(spread_prior["week_key"].max())

            out.loc[
                cur_mask,
                "anchored_margin",
            ] = (
                out.loc[cur_mask, "market_margin"]
                + w
                * out.loc[cur_mask, "spread_model_gap"]
            )

        total_prior = prior.dropna(
            subset=[
                "total_model_gap",
                "total_actual_market_residual",
            ]
        )

        if len(total_prior) >= min_prior_rows:
            w = learn_weight(
                total_prior["total_model_gap"],
                total_prior["total_actual_market_residual"],
            )

            out.loc[
                cur_mask,
                "total_anchor_weight",
            ] = w

            out.loc[
                cur_mask,
                "total_anchor_prior_rows",
            ] = len(total_prior)

            out.loc[
                cur_mask,
                "total_anchor_max_source_week",
            ] = float(total_prior["week_key"].max())

            out.loc[
                cur_mask,
                "anchored_total",
            ] = (
                out.loc[cur_mask, "total_line"]
                + w
                * out.loc[cur_mask, "total_model_gap"]
            )

    return out


def validate_time_safety(d: pd.DataFrame):
    checks = {}

    for market in ["spread", "total"]:
        pred_col = (
            "anchored_margin"
            if market == "spread"
            else "anchored_total"
        )
        src_col = f"{market}_anchor_max_source_week"

        used = d[pred_col].notna()

        bad = d[
            used
            & (
                d[src_col]
                >= d["week_key"]
            )
        ]

        if len(bad):
            raise RuntimeError(
                f"{market} anchor leakage detected."
            )

        checks[
            f"{market}_all_sources_strictly_prior"
        ] = True

    return checks


def metric_summary(d: pd.DataFrame):
    rows = []

    eval_d = d[
        d["anchored_margin"].notna()
        & d["anchored_total"].notna()
    ].copy()

    for market in ["spread", "total"]:
        if market == "spread":
            actual = "actual_margin"
            market_pred = "market_margin"
            base_pred = "pred_margin"
            anchor_pred = "anchored_margin"
            weight = "spread_anchor_weight"
        else:
            actual = "actual_total"
            market_pred = "total_line"
            base_pred = "pred_total"
            anchor_pred = "anchored_total"
            weight = "total_anchor_weight"

        x = eval_d.dropna(
            subset=[
                actual,
                market_pred,
                base_pred,
                anchor_pred,
            ]
        )

        rows.append({
            "market": market,
            "games": len(x),
            "market_mae": mae(x[actual], x[market_pred]),
            "base_model_mae": mae(x[actual], x[base_pred]),
            "anchored_mae": mae(x[actual], x[anchor_pred]),
            "anchor_delta_mae_vs_market": (
                mae(x[actual], x[anchor_pred])
                - mae(x[actual], x[market_pred])
            ),
            "anchor_delta_mae_vs_base": (
                mae(x[actual], x[anchor_pred])
                - mae(x[actual], x[base_pred])
            ),
            "market_rmse": rmse(x[actual], x[market_pred]),
            "base_model_rmse": rmse(x[actual], x[base_pred]),
            "anchored_rmse": rmse(x[actual], x[anchor_pred]),
            "anchor_delta_rmse_vs_market": (
                rmse(x[actual], x[anchor_pred])
                - rmse(x[actual], x[market_pred])
            ),
            "anchor_delta_rmse_vs_base": (
                rmse(x[actual], x[anchor_pred])
                - rmse(x[actual], x[base_pred])
            ),
            "avg_anchor_weight": float(x[weight].mean()),
            "min_anchor_weight": float(x[weight].min()),
            "max_anchor_weight": float(x[weight].max()),
        })

    return pd.DataFrame(rows)


def season_summary(d: pd.DataFrame):
    rows = []

    for season, sd in d.groupby("season"):
        for market in ["spread", "total"]:
            if market == "spread":
                actual = "actual_margin"
                market_pred = "market_margin"
                base_pred = "pred_margin"
                anchor_pred = "anchored_margin"
                weight = "spread_anchor_weight"
            else:
                actual = "actual_total"
                market_pred = "total_line"
                base_pred = "pred_total"
                anchor_pred = "anchored_total"
                weight = "total_anchor_weight"

            x = sd.dropna(
                subset=[
                    actual,
                    market_pred,
                    base_pred,
                    anchor_pred,
                ]
            )

            if x.empty:
                continue

            rows.append({
                "season": int(season),
                "market": market,
                "games": len(x),
                "market_mae": mae(x[actual], x[market_pred]),
                "base_model_mae": mae(x[actual], x[base_pred]),
                "anchored_mae": mae(x[actual], x[anchor_pred]),
                "anchor_delta_mae_vs_market": (
                    mae(x[actual], x[anchor_pred])
                    - mae(x[actual], x[market_pred])
                ),
                "market_rmse": rmse(x[actual], x[market_pred]),
                "base_model_rmse": rmse(x[actual], x[base_pred]),
                "anchored_rmse": rmse(x[actual], x[anchor_pred]),
                "anchor_delta_rmse_vs_market": (
                    rmse(x[actual], x[anchor_pred])
                    - rmse(x[actual], x[market_pred])
                ),
                "avg_anchor_weight": float(x[weight].mean()),
            })

    return pd.DataFrame(rows).sort_values(
        ["market", "season"]
    )


def gap_bucket_summary(d: pd.DataFrame):
    rows = []

    for market in ["spread", "total"]:
        gap_col = (
            "spread_model_gap"
            if market == "spread"
            else "total_model_gap"
        )
        residual_col = (
            "spread_actual_market_residual"
            if market == "spread"
            else "total_actual_market_residual"
        )
        weight_col = f"{market}_anchor_weight"

        x = d.dropna(
            subset=[
                gap_col,
                residual_col,
                weight_col,
            ]
        ).copy()

        x["abs_gap"] = x[gap_col].abs()

        bins = [0, 1, 2, 3, 5, 7.5, 10, np.inf]
        labels = [
            "0-1",
            "1-2",
            "2-3",
            "3-5",
            "5-7.5",
            "7.5-10",
            "10+",
        ]

        x["gap_bucket"] = pd.cut(
            x["abs_gap"],
            bins=bins,
            labels=labels,
            right=False,
        )

        for bucket, g in x.groupby(
            "gap_bucket",
            observed=True,
        ):
            if g.empty:
                continue

            same_direction = (
                np.sign(g[gap_col])
                == np.sign(g[residual_col])
            )

            rows.append({
                "market": market,
                "gap_bucket": str(bucket),
                "games": len(g),
                "avg_abs_model_gap": float(g["abs_gap"].mean()),
                "market_error_same_direction_rate": float(
                    same_direction.mean()
                ),
                "avg_actual_market_residual_when_model_disagrees": float(
                    (
                        np.sign(g[gap_col])
                        * g[residual_col]
                    ).mean()
                ),
                "avg_anchor_weight": float(g[weight_col].mean()),
            })

    return pd.DataFrame(rows)


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
        "--output-dir",
        default="artifacts/market_anchor_v34",
    )
    ap.add_argument(
        "--min-prior-rows",
        type=int,
        default=200,
    )
    ap.add_argument(
        "--first-eval-season",
        type=int,
        default=2023,
    )

    args = ap.parse_args()

    predictions = pd.read_csv(args.predictions)
    market = pd.read_csv(args.market_csv)

    d = prepare(predictions, market)

    anchored = walk_forward_anchor(
        d,
        min_prior_rows=args.min_prior_rows,
        first_eval_season=args.first_eval_season,
    )

    validation = validate_time_safety(anchored)
    overall = metric_summary(anchored)
    by_season = season_summary(anchored)
    buckets = gap_bucket_summary(anchored)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    anchored.to_csv(
        out / "walk_forward_anchored_predictions.csv",
        index=False,
    )
    overall.to_csv(
        out / "overall_metrics.csv",
        index=False,
    )
    by_season.to_csv(
        out / "by_season.csv",
        index=False,
    )
    buckets.to_csv(
        out / "gap_buckets.csv",
        index=False,
    )

    print("\nV3.4 TIME-SAFETY")
    print(validation)

    print("\nV3.4 OVERALL — MARKET vs BASE MODEL vs MARKET-ANCHORED")
    print(overall.to_string(index=False))

    print("\nV3.4 BY SEASON")
    print(by_season.to_string(index=False))

    print("\nV3.4 MODEL-vs-MARKET GAP BUCKETS")
    print(buckets.to_string(index=False))

    print(
        "\nInterpret anchor weight:"
        "\n  0.00 = market only"
        "\n  1.00 = raw model only"
        "\n  0.20 = keep 20% of the model's disagreement with the market"
    )

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
