from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_COLS = {
    "raw": "raw_probability",
    "isotonic_v26": "calibrated_probability",
    "platt_v32": "platt_probability",
}

LAMBDA_GRID = [0.25, 0.50, 0.75, 1.00]
MINIMUM_EDGE = 0.01


def shrink_probability(p, lam):
    return 0.5 + float(lam) * (np.asarray(p, dtype=float) - 0.5)


def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def choose_lambda_by_prior_brier(
    prior: pd.DataFrame,
    probability_col: str,
):
    d = prior[
        prior["win"].notna()
        & prior[probability_col].notna()
    ].copy()

    if d.empty:
        return None, pd.DataFrame()

    rows = []
    y = d["win"].to_numpy(float)
    p = d[probability_col].to_numpy(float)

    for lam in LAMBDA_GRID:
        sp = shrink_probability(p, lam)
        rows.append({
            "lambda": lam,
            "rows": len(d),
            "brier": brier(y, sp),
        })

    table = pd.DataFrame(rows).sort_values(
        ["brier", "lambda"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return float(table.iloc[0]["lambda"]), table


def select_season_bets(
    target: pd.DataFrame,
    probability_col: str,
    lam: float,
):
    d = target.copy()

    d["method_probability"] = pd.to_numeric(
        d[probability_col], errors="coerce"
    )

    d["selected_probability"] = shrink_probability(
        d["method_probability"],
        lam,
    )

    d["selected_edge"] = (
        d["selected_probability"]
        - pd.to_numeric(d["break_even"], errors="coerce")
    )

    d = (
        d.sort_values(
            ["game_id", "market", "selected_edge"],
            ascending=[True, True, False],
        )
        .drop_duplicates(
            ["game_id", "market"],
            keep="first",
        )
        .copy()
    )

    d["bet"] = d["selected_edge"] >= MINIMUM_EDGE
    return d


def run_frozen_validation(df: pd.DataFrame):
    seasons = sorted(
        int(x) for x in df["season"].dropna().unique()
    )

    if len(seasons) < 2:
        raise ValueError(
            "Need at least two seasons for frozen prior-season validation."
        )

    eval_seasons = seasons[1:]

    selected_parts = []
    lambda_rows = []

    for market in ["spread", "total"]:
        md = df[df["market"] == market].copy()

        for method, probability_col in METHOD_COLS.items():
            for season in eval_seasons:
                prior = md[
                    (md["season"] < season)
                    & md["win"].notna()
                ].copy()

                target = md[md["season"] == season].copy()

                if prior.empty or target.empty:
                    continue

                lam, table = choose_lambda_by_prior_brier(
                    prior,
                    probability_col,
                )

                if lam is None:
                    continue

                prior_brier = float(
                    table.iloc[0]["brier"]
                )

                lambda_rows.append({
                    "target_season": season,
                    "market": market,
                    "method": method,
                    "chosen_lambda": lam,
                    "prior_rows": int(table.iloc[0]["rows"]),
                    "prior_best_brier": prior_brier,
                    "prior_seasons": (
                        f"{int(prior['season'].min())}-"
                        f"{int(prior['season'].max())}"
                    ),
                })

                s = select_season_bets(
                    target,
                    probability_col,
                    lam,
                )

                s["method"] = method
                s["chosen_lambda"] = lam
                s["target_season"] = season
                selected_parts.append(s)

    if not selected_parts:
        raise RuntimeError("No frozen validation rows were created.")

    selected = pd.concat(selected_parts, ignore_index=True)
    lambda_table = pd.DataFrame(lambda_rows)

    return selected, lambda_table


def summarize(selected: pd.DataFrame):
    rows = []

    for (market, method), g0 in selected.groupby(
        ["market", "method"]
    ):
        g = g0[g0["bet"]].copy()
        settled = g[
            g["result"].isin(["win", "loss"])
        ].copy()

        units = (
            float(settled["profit"].sum())
            if len(settled)
            else 0.0
        )

        rows.append({
            "market": market,
            "method": method,
            "bets": len(g),
            "settled": len(settled),
            "pushes": int((g["result"] == "push").sum()),
            "wins": int((settled["result"] == "win").sum()),
            "win_rate": (
                float((settled["result"] == "win").mean())
                if len(settled)
                else np.nan
            ),
            "units": units,
            "roi": (
                units / len(settled)
                if len(settled)
                else np.nan
            ),
            "avg_edge": (
                float(g["selected_edge"].mean())
                if len(g)
                else np.nan
            ),
        })

    return pd.DataFrame(rows).sort_values(
        ["market", "method"]
    )


def summarize_by_season(selected: pd.DataFrame):
    rows = []

    for (season, market, method), g0 in selected.groupby(
        ["season", "market", "method"]
    ):
        g = g0[g0["bet"]].copy()
        settled = g[
            g["result"].isin(["win", "loss"])
        ].copy()

        units = (
            float(settled["profit"].sum())
            if len(settled)
            else 0.0
        )

        rows.append({
            "season": int(season),
            "market": market,
            "method": method,
            "chosen_lambda": (
                float(g0["chosen_lambda"].iloc[0])
                if len(g0)
                else np.nan
            ),
            "bets": len(g),
            "settled": len(settled),
            "pushes": int((g["result"] == "push").sum()),
            "win_rate": (
                float((settled["result"] == "win").mean())
                if len(settled)
                else np.nan
            ),
            "units": units,
            "roi": (
                units / len(settled)
                if len(settled)
                else np.nan
            ),
        })

    return pd.DataFrame(rows).sort_values(
        ["market", "method", "season"]
    )


def week_block_bootstrap(
    selected: pd.DataFrame,
    reps=3000,
    seed=33,
):
    rng = np.random.default_rng(seed)
    rows = []

    for (market, method), g0 in selected.groupby(
        ["market", "method"]
    ):
        g = g0[
            g0["bet"]
            & g0["result"].isin(["win", "loss"])
        ].copy()

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
        n = len(blocks)

        for _ in range(reps):
            idx = rng.integers(0, n, size=n)
            units = sum(blocks[i][1] for i in idx)
            bets = sum(blocks[i][2] for i in idx)
            if bets:
                sims.append(units / bets)

        rows.append({
            "market": market,
            "method": method,
            "settled_bets": len(g),
            "week_blocks": len(blocks),
            "roi_point_estimate": float(
                g["profit"].sum() / len(g)
            ),
            "roi_ci_low_95": float(np.quantile(sims, 0.025)),
            "roi_ci_high_95": float(np.quantile(sims, 0.975)),
        })

    return pd.DataFrame(rows).sort_values(
        ["market", "method"]
    )


def calibration_on_eval_seasons(
    selected_source: pd.DataFrame,
    lambda_table: pd.DataFrame,
):
    rows = []

    for _, cfg in lambda_table.iterrows():
        season = int(cfg["target_season"])
        market = cfg["market"]
        method = cfg["method"]
        lam = float(cfg["chosen_lambda"])
        col = METHOD_COLS[method]

        d = selected_source[
            (selected_source["season"] == season)
            & (selected_source["market"] == market)
            & selected_source["win"].notna()
            & selected_source[col].notna()
        ].copy()

        if d.empty:
            continue

        p = shrink_probability(
            d[col].to_numpy(float),
            lam,
        )
        y = d["win"].to_numpy(float)

        rows.append({
            "season": season,
            "market": market,
            "method": method,
            "chosen_lambda": lam,
            "rows": len(d),
            "brier": brier(y, p),
        })

    return pd.DataFrame(rows).sort_values(
        ["market", "method", "season"]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--probabilities",
        default=(
            "artifacts/calibration_v32/"
            "walk_forward_probabilities.csv"
        ),
    )
    ap.add_argument(
        "--output-dir",
        default="artifacts/calibration_v33",
    )
    ap.add_argument(
        "--bootstrap-reps",
        type=int,
        default=3000,
    )
    args = ap.parse_args()

    df = pd.read_csv(args.probabilities)

    required = {
        "game_id",
        "season",
        "week",
        "week_key",
        "market",
        "side",
        "odds",
        "break_even",
        "result",
        "win",
        "profit",
        *METHOD_COLS.values(),
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing v3.2 probability columns: {missing}"
        )

    selected, lambdas = run_frozen_validation(df)
    overall = summarize(selected)
    by_season = summarize_by_season(selected)
    boot = week_block_bootstrap(
        selected,
        reps=args.bootstrap_reps,
    )
    calibration = calibration_on_eval_seasons(
        df,
        lambdas,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lambdas.to_csv(
        out / "chosen_lambdas_by_target_season.csv",
        index=False,
    )
    selected.to_csv(
        out / "frozen_selected_rows.csv",
        index=False,
    )
    overall.to_csv(
        out / "frozen_betting_summary.csv",
        index=False,
    )
    by_season.to_csv(
        out / "frozen_by_season.csv",
        index=False,
    )
    boot.to_csv(
        out / "frozen_week_block_bootstrap_roi.csv",
        index=False,
    )
    calibration.to_csv(
        out / "frozen_calibration_by_season.csv",
        index=False,
    )

    print("\nV3.3 CHOSEN SHRINKAGE — PRIOR BRIER ONLY")
    print(lambdas.to_string(index=False))

    print("\nV3.3 FROZEN FUTURE-SEASON BETTING SUMMARY")
    print(overall.to_string(index=False))

    print("\nV3.3 FROZEN RESULTS BY SEASON")
    print(by_season.to_string(index=False))

    print("\nV3.3 FUTURE-SEASON CALIBRATION")
    print(calibration.to_string(index=False))

    print("\nV3.3 WEEK-BLOCK ROI 95% INTERVALS")
    print(boot.to_string(index=False))

    print(
        "\nDesign note: no ROI was used to choose shrinkage. "
        "Each target season uses only earlier seasons to choose lambda "
        "by Brier score, then freezes it for the entire target season."
    )

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
