from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


EPS = 1e-6

# Freeze the CURRENT live decision rules so this test isolates calibration.
PRODUCTION_RULES = {
    "spread": {
        "lambda": 0.40,
        "minimum_edge": 0.01,
    },
    "total": {
        "lambda": 0.25,
        "minimum_edge": 0.01,
    },
}


def logit(p):
    x = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(x / (1.0 - x))


def log_loss(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return float(
        -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    )


def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def ece_equal_width(y, p, bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0

    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        n = int(mask.sum())
        if not n:
            continue

        ece += (
            n / total
            * abs(float(p[mask].mean()) - float(y[mask].mean()))
        )

    return float(ece)


def fit_platt(prior: pd.DataFrame):
    settled = prior[
        prior["win"].notna()
        & prior["raw_probability"].notna()
    ].copy()

    if len(settled) < 2 or settled["win"].nunique() < 2:
        return None

    x = logit(settled["raw_probability"].to_numpy()).reshape(-1, 1)
    y = settled["win"].astype(int).to_numpy()

    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000,
    )
    model.fit(x, y)
    return model


def add_walk_forward_platt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strict time-safe Platt calibration.

    For week W, only settled rows with week_key < W are used.
    To keep the activation point aligned with the deployed v2.6 calibration,
    Platt remains a raw-probability fallback while deployed
    `calibration_rows` is zero.
    """
    out = df.copy()
    out["platt_probability"] = out["raw_probability"].astype(float)
    out["platt_calibration_rows"] = 0
    out["platt_max_source_week_key"] = np.nan
    out["platt_mode"] = "raw_fallback"

    for market in sorted(out["market"].dropna().unique()):
        market_mask = out["market"].eq(market)
        weeks = sorted(out.loc[market_mask, "week_key"].dropna().unique())

        for wk in weeks:
            cur_mask = market_mask & out["week_key"].eq(wk)
            current = out.loc[cur_mask]

            # Match the deployed calibration's activation timing.
            deployed_rows = pd.to_numeric(
                current["calibration_rows"], errors="coerce"
            ).fillna(0)

            if deployed_rows.max() <= 0:
                continue

            prior = out[
                market_mask
                & (out["week_key"] < wk)
                & out["win"].notna()
                & out["raw_probability"].notna()
            ].copy()

            model = fit_platt(prior)
            if model is None:
                continue

            x = logit(
                out.loc[cur_mask, "raw_probability"].to_numpy()
            ).reshape(-1, 1)

            pred = model.predict_proba(x)[:, 1]

            out.loc[cur_mask, "platt_probability"] = pred
            out.loc[cur_mask, "platt_calibration_rows"] = len(prior)
            out.loc[cur_mask, "platt_max_source_week_key"] = (
                float(prior["week_key"].max())
            )
            out.loc[cur_mask, "platt_mode"] = "platt"

    return out


def validate_time_safety(df: pd.DataFrame):
    used = df["platt_mode"].eq("platt")

    bad = df[
        used
        & (
            pd.to_numeric(
                df["platt_max_source_week_key"], errors="coerce"
            )
            >= pd.to_numeric(df["week_key"], errors="coerce")
        )
    ]

    if len(bad):
        raise RuntimeError(
            "Platt calibration leakage detected: source week was not "
            "strictly earlier than prediction week."
        )

    return {
        "platt_all_source_weeks_strictly_prior": True,
        "platt_rows": int(used.sum()),
        "platt_fallback_rows": int((~used).sum()),
    }


def long_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    mapping = {
        "raw": "raw_probability",
        "isotonic_v26": "calibrated_probability",
        "platt_v32": "platt_probability",
    }

    base_cols = [
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
        "raw_probability",
        "calibration_rows",
    ]

    for method, col in mapping.items():
        d = df[base_cols].copy()
        d["method"] = method
        d["probability"] = pd.to_numeric(df[col], errors="coerce")
        pieces.append(d)

    return pd.concat(pieces, ignore_index=True)


def calibration_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    settled = long_df[
        long_df["win"].notna()
        & long_df["probability"].notna()
    ].copy()

    for (market, method), g in settled.groupby(["market", "method"]):
        y = g["win"].to_numpy(float)
        p = g["probability"].to_numpy(float)

        rows.append({
            "market": market,
            "method": method,
            "rows": len(g),
            "brier": brier(y, p),
            "log_loss": log_loss(y, p),
            "ece_10": ece_equal_width(y, p, bins=10),
            "mean_probability": float(np.mean(p)),
            "actual_win_rate": float(np.mean(y)),
            "probability_std": float(np.std(p)),
            "unique_probability_values": int(
                pd.Series(np.round(p, 8)).nunique()
            ),
            "p10": float(np.quantile(p, 0.10)),
            "p50": float(np.quantile(p, 0.50)),
            "p90": float(np.quantile(p, 0.90)),
            "max_probability": float(np.max(p)),
        })

    return pd.DataFrame(rows).sort_values(["market", "brier"])


def reliability_table(long_df: pd.DataFrame, bins=10) -> pd.DataFrame:
    rows = []

    settled = long_df[
        long_df["win"].notna()
        & long_df["probability"].notna()
    ].copy()

    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    for (market, method), g in settled.groupby(["market", "method"]):
        p = g["probability"].to_numpy(float)
        y = g["win"].to_numpy(float)

        for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if hi == 1.0:
                mask = (p >= lo) & (p <= hi)
            else:
                mask = (p >= lo) & (p < hi)

            n = int(mask.sum())
            if not n:
                continue

            rows.append({
                "market": market,
                "method": method,
                "bin": i + 1,
                "bin_low": lo,
                "bin_high": hi,
                "rows": n,
                "avg_probability": float(np.mean(p[mask])),
                "actual_win_rate": float(np.mean(y[mask])),
                "calibration_gap": float(
                    np.mean(p[mask]) - np.mean(y[mask])
                ),
            })

    return pd.DataFrame(rows)


def compression_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    settled = long_df[
        long_df["win"].notna()
        & long_df["probability"].notna()
    ].copy()

    for market in sorted(settled["market"].unique()):
        m = settled[settled["market"] == market]

        # One copy per method is fine because raw_probability is repeated.
        for cutoff in [0.55, 0.60, 0.65, 0.70]:
            for method in sorted(m["method"].unique()):
                g = m[
                    (m["method"] == method)
                    & (m["raw_probability"] >= cutoff)
                ]
                if g.empty:
                    continue

                rows.append({
                    "market": market,
                    "raw_probability_cutoff": cutoff,
                    "method": method,
                    "rows": len(g),
                    "avg_raw_probability": float(
                        g["raw_probability"].mean()
                    ),
                    "avg_method_probability": float(
                        g["probability"].mean()
                    ),
                    "actual_win_rate": float(g["win"].mean()),
                    "method_minus_actual": float(
                        g["probability"].mean() - g["win"].mean()
                    ),
                })

    return pd.DataFrame(rows)


def select_fixed_rule(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Hold the live decision rule fixed to isolate calibration.

    For each method/game/market:
      1. shrink calibrated probability toward 0.5 using the production lambda
      2. compute edge vs exact historical break-even
      3. keep exactly one side per game/market
      4. bet only if edge >= production minimum edge
    """
    d = long_df.copy()

    d["lambda"] = d["market"].map(
        {k: v["lambda"] for k, v in PRODUCTION_RULES.items()}
    )
    d["minimum_edge"] = d["market"].map(
        {k: v["minimum_edge"] for k, v in PRODUCTION_RULES.items()}
    )

    d["selected_probability"] = (
        0.5 + d["lambda"] * (d["probability"] - 0.5)
    )
    d["selected_edge"] = (
        d["selected_probability"] - d["break_even"]
    )

    best = (
        d.sort_values(
            [
                "method",
                "game_id",
                "market",
                "selected_edge",
            ],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(
            ["method", "game_id", "market"],
            keep="first",
        )
        .copy()
    )

    best["bet"] = (
        best["selected_edge"] >= best["minimum_edge"]
    )

    return best.sort_values(
        ["method", "week_key", "market", "game_id"]
    ).reset_index(drop=True)


def betting_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (market, method), g0 in selected.groupby(["market", "method"]):
        g = g0[g0["bet"]].copy()
        settled = g[g["result"].isin(["win", "loss"])].copy()
        pushes = int((g["result"] == "push").sum())

        units = float(settled["profit"].sum()) if len(settled) else 0.0
        roi = units / len(settled) if len(settled) else np.nan

        rows.append({
            "market": market,
            "method": method,
            "bets": len(g),
            "settled": len(settled),
            "pushes": pushes,
            "wins": int((settled["result"] == "win").sum()),
            "win_rate": (
                float((settled["result"] == "win").mean())
                if len(settled)
                else np.nan
            ),
            "units": units,
            "roi": roi,
            "avg_edge": (
                float(g["selected_edge"].mean())
                if len(g)
                else np.nan
            ),
            "avg_selected_probability": (
                float(g["selected_probability"].mean())
                if len(g)
                else np.nan
            ),
        })

    return pd.DataFrame(rows).sort_values(["market", "method"])


def betting_by_season(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (season, market, method), g0 in selected.groupby(
        ["season", "market", "method"]
    ):
        g = g0[g0["bet"]].copy()
        settled = g[g["result"].isin(["win", "loss"])].copy()

        units = float(settled["profit"].sum()) if len(settled) else 0.0

        rows.append({
            "season": season,
            "market": market,
            "method": method,
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


def week_block_bootstrap_roi(
    selected: pd.DataFrame,
    reps=2000,
    seed=32,
) -> pd.DataFrame:
    """
    Resample week blocks, not individual bets, so same-week correlation is
    retained. This is descriptive uncertainty, not proof of profitability.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for (market, method), g0 in selected.groupby(["market", "method"]):
        g = g0[
            g0["bet"]
            & g0["result"].isin(["win", "loss"])
        ].copy()

        if g.empty:
            continue

        by_week = []
        for wk, x in g.groupby("week_key"):
            by_week.append(
                (
                    wk,
                    float(x["profit"].sum()),
                    int(len(x)),
                )
            )

        n_weeks = len(by_week)
        if n_weeks < 2:
            continue

        rois = []
        for _ in range(reps):
            idx = rng.integers(0, n_weeks, size=n_weeks)
            units = sum(by_week[i][1] for i in idx)
            bets = sum(by_week[i][2] for i in idx)
            if bets:
                rois.append(units / bets)

        if not rois:
            continue

        rows.append({
            "market": market,
            "method": method,
            "settled_bets": len(g),
            "week_blocks": n_weeks,
            "bootstrap_reps": len(rois),
            "roi_point_estimate": float(g["profit"].sum() / len(g)),
            "roi_ci_low_95": float(np.quantile(rois, 0.025)),
            "roi_ci_high_95": float(np.quantile(rois, 0.975)),
        })

    return pd.DataFrame(rows).sort_values(["market", "method"])


def compare_deployed_isotonic_to_raw(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for market, g in df.groupby("market"):
        active = g[
            pd.to_numeric(g["calibration_rows"], errors="coerce").fillna(0) > 0
        ].copy()

        if active.empty:
            continue

        delta = (
            pd.to_numeric(active["calibrated_probability"], errors="coerce")
            - pd.to_numeric(active["raw_probability"], errors="coerce")
        )

        rows.append({
            "market": market,
            "active_rows": len(active),
            "mean_abs_probability_change": float(delta.abs().mean()),
            "median_abs_probability_change": float(delta.abs().median()),
            "max_abs_probability_change": float(delta.abs().max()),
            "share_moved_toward_50pct": float(
                (
                    (
                        active["calibrated_probability"] - 0.5
                    ).abs()
                    <
                    (
                        active["raw_probability"] - 0.5
                    ).abs()
                ).mean()
            ),
        })

    return pd.DataFrame(rows)


def validate_input(df: pd.DataFrame):
    required = {
        "game_id",
        "season",
        "week",
        "week_key",
        "market",
        "side",
        "raw_probability",
        "odds",
        "break_even",
        "result",
        "win",
        "profit",
        "calibrated_probability",
        "calibration_rows",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    markets = set(df["market"].dropna().unique())
    unexpected = markets - set(PRODUCTION_RULES)
    if unexpected:
        raise ValueError(f"Unexpected markets: {sorted(unexpected)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--calibrated-sides",
        default="artifacts/calibration_v28_raw/calibrated_sides.csv",
    )
    ap.add_argument(
        "--output-dir",
        default="artifacts/calibration_v32",
    )
    ap.add_argument(
        "--bootstrap-reps",
        type=int,
        default=2000,
    )
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.calibrated_sides)
    validate_input(df)

    df["week_key"] = pd.to_numeric(df["week_key"], errors="raise")
    df["raw_probability"] = pd.to_numeric(
        df["raw_probability"], errors="coerce"
    )
    df["calibrated_probability"] = pd.to_numeric(
        df["calibrated_probability"], errors="coerce"
    )
    df["break_even"] = pd.to_numeric(df["break_even"], errors="coerce")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")

    v = add_walk_forward_platt(df)
    audit = validate_time_safety(v)

    long_df = long_probabilities(v)
    metrics = calibration_metrics(long_df)
    reliability = reliability_table(long_df)
    compression = compression_summary(long_df)
    selected = select_fixed_rule(long_df)
    bet_summary = betting_summary(selected)
    season = betting_by_season(selected)
    boot = week_block_bootstrap_roi(
        selected,
        reps=args.bootstrap_reps,
    )
    iso_compression = compare_deployed_isotonic_to_raw(v)

    v.to_csv(
        outdir / "walk_forward_probabilities.csv",
        index=False,
    )
    metrics.to_csv(
        outdir / "calibration_metrics.csv",
        index=False,
    )
    reliability.to_csv(
        outdir / "reliability_bins.csv",
        index=False,
    )
    compression.to_csv(
        outdir / "high_raw_probability_compression.csv",
        index=False,
    )
    selected.to_csv(
        outdir / "fixed_rule_selected_rows.csv",
        index=False,
    )
    bet_summary.to_csv(
        outdir / "fixed_rule_betting_summary.csv",
        index=False,
    )
    season.to_csv(
        outdir / "fixed_rule_by_season.csv",
        index=False,
    )
    boot.to_csv(
        outdir / "week_block_bootstrap_roi.csv",
        index=False,
    )
    iso_compression.to_csv(
        outdir / "isotonic_compression_summary.csv",
        index=False,
    )

    report = {
        "validation": audit,
        "design": {
            "methods": [
                "raw",
                "isotonic_v26",
                "platt_v32",
            ],
            "platt_input": "logit(raw_probability)",
            "platt_training": "strict prior weeks within market only",
            "deployed_isotonic_source": (
                "existing calibrated_probability from "
                "calibrated_sides.csv"
            ),
            "decision_rules_frozen": PRODUCTION_RULES,
            "selector": (
                "one side per game/market, maximum selected edge, "
                "then minimum-edge gate"
            ),
            "bootstrap": (
                "week-block resampling; descriptive uncertainty only"
            ),
            "auto_promotion": False,
        },
        "rows": int(len(df)),
        "settled_rows": int(df["win"].notna().sum()),
        "markets": sorted(df["market"].dropna().unique().tolist()),
    }

    (outdir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nV3.2 VALIDATION")
    print(json.dumps(audit, indent=2))

    print("\nCALIBRATION METRICS — LOWER IS BETTER")
    print(metrics.to_string(index=False))

    print("\nDEPLOYED ISOTONIC COMPRESSION")
    print(iso_compression.to_string(index=False))

    print("\nFIXED CURRENT LIVE RULES — BETTING SUMMARY")
    print(bet_summary.to_string(index=False))

    print("\nFIXED CURRENT LIVE RULES — BY SEASON")
    print(season.to_string(index=False))

    print("\nWEEK-BLOCK BOOTSTRAP ROI 95% INTERVALS")
    print(boot.to_string(index=False))

    print(
        "\nNo method is automatically promoted. "
        "Compare calibration quality, season stability, bet count, "
        "and ROI uncertainty together."
    )

    print(f"\nSaved diagnostics to: {outdir}")


if __name__ == "__main__":
    main()
