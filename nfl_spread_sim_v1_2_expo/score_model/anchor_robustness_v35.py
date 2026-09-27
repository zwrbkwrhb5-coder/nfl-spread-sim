from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .market_anchor_v34 import prepare, learn_weight


def mae(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(y - pred)))


def rmse(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def frozen_season_anchor(
    d: pd.DataFrame,
    first_eval_season: int = 2023,
    min_prior_rows: int = 200,
):
    """
    For target season S:
      - learn one spread anchor weight from seasons < S
      - learn one total anchor weight from seasons < S
      - freeze each weight for ALL of season S

    No within-season updating.
    """
    parts = []
    config = []

    seasons = sorted(
        int(x) for x in d["season"].dropna().unique()
    )

    for season in seasons:
        if season < first_eval_season:
            continue

        prior = d[d["season"] < season].copy()
        target = d[d["season"] == season].copy()

        if target.empty:
            continue

        spread_prior = prior.dropna(
            subset=[
                "spread_model_gap",
                "spread_actual_market_residual",
            ]
        )
        total_prior = prior.dropna(
            subset=[
                "total_model_gap",
                "total_actual_market_residual",
            ]
        )

        spread_w = np.nan
        total_w = np.nan

        if len(spread_prior) >= min_prior_rows:
            spread_w = learn_weight(
                spread_prior["spread_model_gap"],
                spread_prior["spread_actual_market_residual"],
            )

        if len(total_prior) >= min_prior_rows:
            total_w = learn_weight(
                total_prior["total_model_gap"],
                total_prior["total_actual_market_residual"],
            )

        t = target.copy()

        t["frozen_spread_weight"] = spread_w
        t["frozen_total_weight"] = total_w

        t["frozen_anchored_margin"] = (
            t["market_margin"]
            + spread_w * t["spread_model_gap"]
            if np.isfinite(spread_w)
            else np.nan
        )

        t["frozen_anchored_total"] = (
            t["total_line"]
            + total_w * t["total_model_gap"]
            if np.isfinite(total_w)
            else np.nan
        )

        t["anchor_source_max_season"] = (
            int(prior["season"].max())
            if not prior.empty
            else np.nan
        )

        parts.append(t)

        config.append({
            "target_season": season,
            "spread_weight": spread_w,
            "spread_prior_rows": len(spread_prior),
            "total_weight": total_w,
            "total_prior_rows": len(total_prior),
            "source_max_season": (
                int(prior["season"].max())
                if not prior.empty
                else np.nan
            ),
        })

    if not parts:
        raise RuntimeError("No frozen-season evaluation rows created.")

    return (
        pd.concat(parts, ignore_index=True),
        pd.DataFrame(config),
    )


def validate_frozen(d: pd.DataFrame):
    bad = d[
        pd.notna(d["anchor_source_max_season"])
        & (
            pd.to_numeric(
                d["anchor_source_max_season"],
                errors="coerce",
            )
            >= pd.to_numeric(
                d["season"],
                errors="coerce",
            )
        )
    ]

    if len(bad):
        raise RuntimeError(
            "Frozen anchor leakage: source season is not prior."
        )

    return {
        "all_anchor_source_seasons_strictly_prior": True,
        "rows": int(len(d)),
    }


def metric_rows(d: pd.DataFrame):
    rows = []

    specs = {
        "spread": {
            "actual": "actual_margin",
            "market": "market_margin",
            "base": "pred_margin",
            "anchor": "frozen_anchored_margin",
            "weight": "frozen_spread_weight",
        },
        "total": {
            "actual": "actual_total",
            "market": "total_line",
            "base": "pred_total",
            "anchor": "frozen_anchored_total",
            "weight": "frozen_total_weight",
        },
    }

    for market, s in specs.items():
        x = d.dropna(
            subset=[
                s["actual"],
                s["market"],
                s["base"],
                s["anchor"],
            ]
        )

        rows.append({
            "market": market,
            "games": len(x),
            "market_mae": mae(x[s["actual"]], x[s["market"]]),
            "base_mae": mae(x[s["actual"]], x[s["base"]]),
            "anchor_mae": mae(x[s["actual"]], x[s["anchor"]]),
            "anchor_minus_market_mae": (
                mae(x[s["actual"]], x[s["anchor"]])
                - mae(x[s["actual"]], x[s["market"]])
            ),
            "anchor_minus_base_mae": (
                mae(x[s["actual"]], x[s["anchor"]])
                - mae(x[s["actual"]], x[s["base"]])
            ),
            "market_rmse": rmse(x[s["actual"]], x[s["market"]]),
            "base_rmse": rmse(x[s["actual"]], x[s["base"]]),
            "anchor_rmse": rmse(x[s["actual"]], x[s["anchor"]]),
            "anchor_minus_market_rmse": (
                rmse(x[s["actual"]], x[s["anchor"]])
                - rmse(x[s["actual"]], x[s["market"]])
            ),
            "anchor_minus_base_rmse": (
                rmse(x[s["actual"]], x[s["anchor"]])
                - rmse(x[s["actual"]], x[s["base"]])
            ),
            "avg_frozen_weight": float(x[s["weight"]].mean()),
        })

    return pd.DataFrame(rows)


def by_season_rows(d: pd.DataFrame):
    rows = []

    specs = {
        "spread": {
            "actual": "actual_margin",
            "market": "market_margin",
            "base": "pred_margin",
            "anchor": "frozen_anchored_margin",
            "weight": "frozen_spread_weight",
        },
        "total": {
            "actual": "actual_total",
            "market": "total_line",
            "base": "pred_total",
            "anchor": "frozen_anchored_total",
            "weight": "frozen_total_weight",
        },
    }

    for season, sd in d.groupby("season"):
        for market, s in specs.items():
            x = sd.dropna(
                subset=[
                    s["actual"],
                    s["market"],
                    s["base"],
                    s["anchor"],
                ]
            )

            if x.empty:
                continue

            rows.append({
                "season": int(season),
                "market": market,
                "games": len(x),
                "frozen_weight": float(x[s["weight"]].iloc[0]),
                "market_mae": mae(x[s["actual"]], x[s["market"]]),
                "base_mae": mae(x[s["actual"]], x[s["base"]]),
                "anchor_mae": mae(x[s["actual"]], x[s["anchor"]]),
                "anchor_minus_market_mae": (
                    mae(x[s["actual"]], x[s["anchor"]])
                    - mae(x[s["actual"]], x[s["market"]])
                ),
                "market_rmse": rmse(x[s["actual"]], x[s["market"]]),
                "base_rmse": rmse(x[s["actual"]], x[s["base"]]),
                "anchor_rmse": rmse(x[s["actual"]], x[s["anchor"]]),
                "anchor_minus_market_rmse": (
                    rmse(x[s["actual"]], x[s["anchor"]])
                    - rmse(x[s["actual"]], x[s["market"]])
                ),
            })

    return pd.DataFrame(rows).sort_values(
        ["market", "season"]
    )


def paired_week_bootstrap(
    d: pd.DataFrame,
    reps: int = 5000,
    seed: int = 35,
):
    """
    Bootstrap whole NFL weeks.

    Negative delta means the anchor is better than the comparison.
    """
    rng = np.random.default_rng(seed)
    rows = []

    specs = {
        "spread": {
            "actual": "actual_margin",
            "market": "market_margin",
            "base": "pred_margin",
            "anchor": "frozen_anchored_margin",
        },
        "total": {
            "actual": "actual_total",
            "market": "total_line",
            "base": "pred_total",
            "anchor": "frozen_anchored_total",
        },
    }

    for market, s in specs.items():
        x = d.dropna(
            subset=[
                "week_key",
                s["actual"],
                s["market"],
                s["base"],
                s["anchor"],
            ]
        ).copy()

        week_groups = {
            wk: g.index.to_numpy()
            for wk, g in x.groupby("week_key")
        }
        weeks = list(week_groups)

        if len(weeks) < 2:
            continue

        sims = {
            "mae_vs_market": [],
            "rmse_vs_market": [],
            "mae_vs_base": [],
            "rmse_vs_base": [],
        }

        for _ in range(reps):
            sampled = rng.integers(
                0,
                len(weeks),
                size=len(weeks),
            )

            idx = np.concatenate([
                week_groups[weeks[i]]
                for i in sampled
            ])

            b = x.loc[idx]

            a = b[s["actual"]].to_numpy(float)
            anc = b[s["anchor"]].to_numpy(float)
            mar = b[s["market"]].to_numpy(float)
            base = b[s["base"]].to_numpy(float)

            sims["mae_vs_market"].append(
                mae(a, anc) - mae(a, mar)
            )
            sims["rmse_vs_market"].append(
                rmse(a, anc) - rmse(a, mar)
            )
            sims["mae_vs_base"].append(
                mae(a, anc) - mae(a, base)
            )
            sims["rmse_vs_base"].append(
                rmse(a, anc) - rmse(a, base)
            )

        for comparison in [
            "mae_vs_market",
            "rmse_vs_market",
            "mae_vs_base",
            "rmse_vs_base",
        ]:
            vals = np.asarray(
                sims[comparison],
                dtype=float,
            )

            if comparison == "mae_vs_market":
                point = (
                    mae(x[s["actual"]], x[s["anchor"]])
                    - mae(x[s["actual"]], x[s["market"]])
                )
            elif comparison == "rmse_vs_market":
                point = (
                    rmse(x[s["actual"]], x[s["anchor"]])
                    - rmse(x[s["actual"]], x[s["market"]])
                )
            elif comparison == "mae_vs_base":
                point = (
                    mae(x[s["actual"]], x[s["anchor"]])
                    - mae(x[s["actual"]], x[s["base"]])
                )
            else:
                point = (
                    rmse(x[s["actual"]], x[s["anchor"]])
                    - rmse(x[s["actual"]], x[s["base"]])
                )

            rows.append({
                "market": market,
                "comparison": comparison,
                "games": len(x),
                "week_blocks": len(weeks),
                "bootstrap_reps": reps,
                "point_delta": point,
                "ci_low_95": float(np.quantile(vals, 0.025)),
                "ci_high_95": float(np.quantile(vals, 0.975)),
                "probability_anchor_better": float(
                    np.mean(vals < 0.0)
                ),
            })

    return pd.DataFrame(rows)


def exploratory_gap_report(d: pd.DataFrame):
    """
    Exploratory only. The >=5 threshold was observed after inspecting v3.4,
    so this output MUST NOT be treated as an untouched validation.
    """
    rows = []

    for market in ["spread", "total"]:
        gap = (
            "spread_model_gap"
            if market == "spread"
            else "total_model_gap"
        )
        residual = (
            "spread_actual_market_residual"
            if market == "spread"
            else "total_actual_market_residual"
        )

        x = d.dropna(
            subset=[gap, residual]
        ).copy()

        x["large_gap"] = x[gap].abs() >= 5.0

        for large, g in x.groupby("large_gap"):
            signed = (
                np.sign(g[gap])
                * g[residual]
            )

            rows.append({
                "market": market,
                "gap_group": "5+" if large else "<5",
                "games": len(g),
                "same_direction_rate": float(
                    (
                        np.sign(g[gap])
                        == np.sign(g[residual])
                    ).mean()
                ),
                "avg_signed_actual_market_residual": float(
                    signed.mean()
                ),
                "median_signed_actual_market_residual": float(
                    signed.median()
                ),
                "note": (
                    "EXPLORATORY: threshold chosen after v3.4 inspection"
                ),
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
        default="artifacts/anchor_robustness_v35",
    )
    ap.add_argument(
        "--first-eval-season",
        type=int,
        default=2023,
    )
    ap.add_argument(
        "--min-prior-rows",
        type=int,
        default=200,
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

    frozen, weights = frozen_season_anchor(
        prepared,
        first_eval_season=args.first_eval_season,
        min_prior_rows=args.min_prior_rows,
    )

    validation = validate_frozen(frozen)
    overall = metric_rows(frozen)
    by_season = by_season_rows(frozen)
    boot = paired_week_bootstrap(
        frozen,
        reps=args.bootstrap_reps,
    )
    exploratory = exploratory_gap_report(frozen)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frozen.to_csv(
        out / "frozen_season_predictions.csv",
        index=False,
    )
    weights.to_csv(
        out / "frozen_weights.csv",
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
    boot.to_csv(
        out / "paired_week_bootstrap.csv",
        index=False,
    )
    exploratory.to_csv(
        out / "exploratory_gap_5plus.csv",
        index=False,
    )

    print("\nV3.5 VALIDATION")
    print(validation)

    print("\nV3.5 FROZEN SEASON WEIGHTS")
    print(weights.to_string(index=False))

    print("\nV3.5 OVERALL")
    print(overall.to_string(index=False))

    print("\nV3.5 BY SEASON")
    print(by_season.to_string(index=False))

    print("\nV3.5 PAIRED WEEK-BLOCK BOOTSTRAP")
    print(boot.to_string(index=False))

    print("\nV3.5 EXPLORATORY 5+ GAP CHECK")
    print(exploratory.to_string(index=False))

    print(
        "\nInterpretation: negative bootstrap deltas favor the anchor. "
        "The 5+ gap section is exploratory only and must not be used as a "
        "production filter without a new forward/frozen validation."
    )

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
