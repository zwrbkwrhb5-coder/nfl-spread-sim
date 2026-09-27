from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def mae(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(y - pred)))


def rmse(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def prepare(predictions: pd.DataFrame, market: pd.DataFrame):
    p = predictions.copy()
    m = market.copy()

    keep = [
        c for c in [
            "game_id",
            "home_spread",
            "total_line",
        ]
        if c in m.columns
    ]

    d = p.merge(
        m[keep].drop_duplicates("game_id"),
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
        raise ValueError(f"Missing required columns: {missing}")

    if "week_key" not in d.columns:
        d["week_key"] = (
            pd.to_numeric(d["season"], errors="raise") * 100
            + pd.to_numeric(d["week"], errors="raise")
        )

    for c in [
        "actual_margin",
        "actual_total",
        "pred_margin",
        "pred_total",
        "home_spread",
        "total_line",
        "week_key",
    ]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["market_margin"] = -d["home_spread"]

    return d.sort_values(
        ["season", "week", "game_id"]
    ).reset_index(drop=True)


def fit_market_only(train, market_col, actual_col):
    x = train[[market_col]].to_numpy(float)
    y = train[actual_col].to_numpy(float)
    model = LinearRegression()
    model.fit(x, y)
    return model


def fit_stack(train, market_col, model_col, actual_col):
    x = train[[market_col, model_col]].to_numpy(float)
    y = train[actual_col].to_numpy(float)
    model = LinearRegression()
    model.fit(x, y)
    return model


def frozen_season_predictions(
    d: pd.DataFrame,
    first_eval_season=2023,
    min_prior_rows=200,
):
    parts = []
    coefficients = []

    specs = {
        "spread": {
            "market_col": "market_margin",
            "model_col": "pred_margin",
            "actual_col": "actual_margin",
            "market_cal_pred": "spread_market_calibrated",
            "stack_pred": "spread_stacked",
        },
        "total": {
            "market_col": "total_line",
            "model_col": "pred_total",
            "actual_col": "actual_total",
            "market_cal_pred": "total_market_calibrated",
            "stack_pred": "total_stacked",
        },
    }

    seasons = sorted(int(s) for s in d["season"].dropna().unique())

    for season in seasons:
        if season < first_eval_season:
            continue

        prior_all = d[d["season"] < season].copy()
        target = d[d["season"] == season].copy()

        if target.empty:
            continue

        t = target.copy()
        t["stack_source_max_season"] = (
            int(prior_all["season"].max())
            if not prior_all.empty
            else np.nan
        )

        for market, s in specs.items():
            prior = prior_all.dropna(
                subset=[
                    s["market_col"],
                    s["model_col"],
                    s["actual_col"],
                ]
            ).copy()

            if len(prior) < min_prior_rows:
                t[s["market_cal_pred"]] = np.nan
                t[s["stack_pred"]] = np.nan
                continue

            market_model = fit_market_only(
                prior,
                s["market_col"],
                s["actual_col"],
            )
            stack_model = fit_stack(
                prior,
                s["market_col"],
                s["model_col"],
                s["actual_col"],
            )

            target_ready = t.dropna(
                subset=[
                    s["market_col"],
                    s["model_col"],
                ]
            )

            market_pred = market_model.predict(
                target_ready[[s["market_col"]]].to_numpy(float)
            )
            stack_pred = stack_model.predict(
                target_ready[
                    [s["market_col"], s["model_col"]]
                ].to_numpy(float)
            )

            t[s["market_cal_pred"]] = np.nan
            t[s["stack_pred"]] = np.nan

            t.loc[
                target_ready.index,
                s["market_cal_pred"],
            ] = market_pred

            t.loc[
                target_ready.index,
                s["stack_pred"],
            ] = stack_pred

            coefficients.append({
                "target_season": season,
                "market": market,
                "prior_rows": len(prior),
                "source_max_season": int(prior["season"].max()),
                "market_only_intercept": float(
                    market_model.intercept_
                ),
                "market_only_market_coef": float(
                    market_model.coef_[0]
                ),
                "stack_intercept": float(
                    stack_model.intercept_
                ),
                "stack_market_coef": float(
                    stack_model.coef_[0]
                ),
                "stack_model_coef": float(
                    stack_model.coef_[1]
                ),
            })

        parts.append(t)

    if not parts:
        raise RuntimeError("No evaluation seasons created.")

    return pd.concat(parts, ignore_index=True), pd.DataFrame(coefficients)


def validate_no_future(coefficients: pd.DataFrame):
    bad = coefficients[
        coefficients["source_max_season"]
        >= coefficients["target_season"]
    ]
    if len(bad):
        raise RuntimeError("Future-season leakage detected.")
    return {
        "all_training_seasons_strictly_prior": True,
        "config_rows": int(len(coefficients)),
    }


def metric_summary(pred: pd.DataFrame):
    rows = []

    specs = {
        "spread": {
            "actual": "actual_margin",
            "market": "market_margin",
            "market_cal": "spread_market_calibrated",
            "base": "pred_margin",
            "stack": "spread_stacked",
        },
        "total": {
            "actual": "actual_total",
            "market": "total_line",
            "market_cal": "total_market_calibrated",
            "base": "pred_total",
            "stack": "total_stacked",
        },
    }

    for market, s in specs.items():
        x = pred.dropna(subset=list(s.values())).copy()
        if x.empty:
            continue

        vals = {
            "market_mae": mae(x[s["actual"]], x[s["market"]]),
            "market_calibrated_mae": mae(
                x[s["actual"]], x[s["market_cal"]]
            ),
            "base_model_mae": mae(x[s["actual"]], x[s["base"]]),
            "stack_mae": mae(x[s["actual"]], x[s["stack"]]),
            "market_rmse": rmse(x[s["actual"]], x[s["market"]]),
            "market_calibrated_rmse": rmse(
                x[s["actual"]], x[s["market_cal"]]
            ),
            "base_model_rmse": rmse(x[s["actual"]], x[s["base"]]),
            "stack_rmse": rmse(x[s["actual"]], x[s["stack"]]),
        }

        rows.append({
            "market": market,
            "games": len(x),
            **vals,
            "stack_minus_market_mae": (
                vals["stack_mae"] - vals["market_mae"]
            ),
            "stack_minus_market_calibrated_mae": (
                vals["stack_mae"]
                - vals["market_calibrated_mae"]
            ),
            "stack_minus_base_mae": (
                vals["stack_mae"] - vals["base_model_mae"]
            ),
            "stack_minus_market_rmse": (
                vals["stack_rmse"] - vals["market_rmse"]
            ),
            "stack_minus_market_calibrated_rmse": (
                vals["stack_rmse"]
                - vals["market_calibrated_rmse"]
            ),
            "stack_minus_base_rmse": (
                vals["stack_rmse"] - vals["base_model_rmse"]
            ),
        })

    return pd.DataFrame(rows)


def by_season_summary(pred: pd.DataFrame):
    rows = []

    specs = {
        "spread": {
            "actual": "actual_margin",
            "market": "market_margin",
            "market_cal": "spread_market_calibrated",
            "base": "pred_margin",
            "stack": "spread_stacked",
        },
        "total": {
            "actual": "actual_total",
            "market": "total_line",
            "market_cal": "total_market_calibrated",
            "base": "pred_total",
            "stack": "total_stacked",
        },
    }

    for season, sd in pred.groupby("season"):
        for market, s in specs.items():
            x = sd.dropna(subset=list(s.values())).copy()
            if x.empty:
                continue

            mm = mae(x[s["actual"]], x[s["market"]])
            mc = mae(x[s["actual"]], x[s["market_cal"]])
            bm = mae(x[s["actual"]], x[s["base"]])
            sm = mae(x[s["actual"]], x[s["stack"]])

            mr = rmse(x[s["actual"]], x[s["market"]])
            mcr = rmse(x[s["actual"]], x[s["market_cal"]])
            br = rmse(x[s["actual"]], x[s["base"]])
            sr = rmse(x[s["actual"]], x[s["stack"]])

            rows.append({
                "season": int(season),
                "market": market,
                "games": len(x),
                "market_mae": mm,
                "market_calibrated_mae": mc,
                "base_model_mae": bm,
                "stack_mae": sm,
                "stack_minus_market_mae": sm - mm,
                "stack_minus_market_calibrated_mae": sm - mc,
                "market_rmse": mr,
                "market_calibrated_rmse": mcr,
                "base_model_rmse": br,
                "stack_rmse": sr,
                "stack_minus_market_rmse": sr - mr,
                "stack_minus_market_calibrated_rmse": sr - mcr,
            })

    return pd.DataFrame(rows).sort_values(["market", "season"])


def paired_week_bootstrap(
    pred: pd.DataFrame,
    reps=5000,
    seed=37,
):
    rng = np.random.default_rng(seed)
    rows = []

    specs = {
        "spread": {
            "actual": "actual_margin",
            "market": "market_margin",
            "market_cal": "spread_market_calibrated",
            "base": "pred_margin",
            "stack": "spread_stacked",
        },
        "total": {
            "actual": "actual_total",
            "market": "total_line",
            "market_cal": "total_market_calibrated",
            "base": "pred_total",
            "stack": "total_stacked",
        },
    }

    for market, s in specs.items():
        x = pred.dropna(
            subset=["week_key"] + list(s.values())
        ).copy()

        week_groups = {
            wk: g.index.to_numpy()
            for wk, g in x.groupby("week_key")
        }
        weeks = list(week_groups)

        comparisons = [
            ("mae_vs_market", s["market"], "mae"),
            (
                "mae_vs_market_calibrated",
                s["market_cal"],
                "mae",
            ),
            ("mae_vs_base", s["base"], "mae"),
            ("rmse_vs_market", s["market"], "rmse"),
            (
                "rmse_vs_market_calibrated",
                s["market_cal"],
                "rmse",
            ),
            ("rmse_vs_base", s["base"], "rmse"),
        ]

        for label, compare_col, metric in comparisons:
            sims = []

            for _ in range(reps):
                sampled = rng.integers(
                    0, len(weeks), size=len(weeks)
                )
                idx = np.concatenate(
                    [week_groups[weeks[i]] for i in sampled]
                )
                b = x.loc[idx]

                if metric == "mae":
                    delta = (
                        mae(b[s["actual"]], b[s["stack"]])
                        - mae(b[s["actual"]], b[compare_col])
                    )
                else:
                    delta = (
                        rmse(b[s["actual"]], b[s["stack"]])
                        - rmse(b[s["actual"]], b[compare_col])
                    )
                sims.append(delta)

            if metric == "mae":
                point = (
                    mae(x[s["actual"]], x[s["stack"]])
                    - mae(x[s["actual"]], x[compare_col])
                )
            else:
                point = (
                    rmse(x[s["actual"]], x[s["stack"]])
                    - rmse(x[s["actual"]], x[compare_col])
                )

            sims = np.asarray(sims, dtype=float)

            rows.append({
                "market": market,
                "comparison": label,
                "games": len(x),
                "week_blocks": len(weeks),
                "bootstrap_reps": reps,
                "point_delta": point,
                "ci_low_95": float(np.quantile(sims, 0.025)),
                "ci_high_95": float(np.quantile(sims, 0.975)),
                "probability_stack_better": float(
                    np.mean(sims < 0.0)
                ),
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions",
        default="artifacts/dual_market/oos_qb_raw_context_v28.csv",
    )
    ap.add_argument(
        "--market-csv",
        default="data/historical_market.csv",
    )
    ap.add_argument(
        "--output-dir",
        default="artifacts/market_stack_v37",
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

    p = pd.read_csv(args.predictions)
    m = pd.read_csv(args.market_csv)

    d = prepare(p, m)
    pred, coef = frozen_season_predictions(
        d,
        first_eval_season=args.first_eval_season,
        min_prior_rows=args.min_prior_rows,
    )

    audit = validate_no_future(coef)
    overall = metric_summary(pred)
    by_season = by_season_summary(pred)
    boot = paired_week_bootstrap(
        pred,
        reps=args.bootstrap_reps,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pred.to_csv(out / "frozen_predictions.csv", index=False)
    coef.to_csv(out / "coefficients.csv", index=False)
    overall.to_csv(out / "overall_metrics.csv", index=False)
    by_season.to_csv(out / "by_season.csv", index=False)
    boot.to_csv(out / "paired_week_bootstrap.csv", index=False)

    print("\nV3.7 TIME-SAFETY")
    print(audit)

    print("\nV3.7 FROZEN COEFFICIENTS")
    print(coef.to_string(index=False))

    print("\nV3.7 OVERALL")
    print(overall.to_string(index=False))

    print("\nV3.7 BY SEASON")
    print(by_season.to_string(index=False))

    print("\nV3.7 PAIRED WEEK-BLOCK BOOTSTRAP")
    print(boot.to_string(index=False))

    print(
        "\nInterpretation: negative deltas favor the stack. "
        "The most important comparison is stack vs market_calibrated, "
        "because that isolates whether OUR model adds information beyond "
        "a simple historical correction to the sportsbook line."
    )

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
