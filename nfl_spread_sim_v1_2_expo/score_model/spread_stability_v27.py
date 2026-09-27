from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_LAMBDAS = [0.10, 0.25, 0.40, 0.55, 0.70]
DEFAULT_THRESHOLDS = [0.00, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
DEFAULT_WINDOWS_WEEKS = [26, 52, 78, 104, 9999]


def brier_score(prob, win):
    p = np.asarray(prob, dtype=float)
    y = np.asarray(win, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean((p[mask] - y[mask]) ** 2))


def lower_confidence_roi(profits, z=1.0):
    """
    Conservative ROI score:
      mean profit - z * standard error

    z=1 is intentionally moderate; this is a ranking score, not a formal CI.
    """
    x = np.asarray(profits, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return -np.inf
    mean = float(np.mean(x))
    if len(x) == 1:
        return mean - 1.0
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return mean - z * se


def segment_stability_score(bets: pd.DataFrame):
    """
    Penalize configurations whose ROI is concentrated in a single season.

    Returns:
      season_mean_roi
      season_std_roi
      worst_season_roi
      stability_score

    Stability score rewards average ROI but penalizes season volatility
    and a negative worst season.
    """
    if bets.empty:
        return {
            "season_mean_roi": np.nan,
            "season_std_roi": np.nan,
            "worst_season_roi": np.nan,
            "stability_score": -np.inf,
        }

    season_rois = []
    for _, g in bets.groupby("season"):
        if len(g) < 15:
            continue
        season_rois.append(float(g["profit"].mean()))

    if not season_rois:
        return {
            "season_mean_roi": float(bets["profit"].mean()),
            "season_std_roi": 0.0,
            "worst_season_roi": float(bets["profit"].mean()),
            "stability_score": float(bets["profit"].mean()),
        }

    arr = np.asarray(season_rois, dtype=float)
    mean_roi = float(np.mean(arr))
    std_roi = float(np.std(arr, ddof=0))
    worst = float(np.min(arr))

    # Penalize volatility and especially a very bad worst season.
    score = mean_roi - 0.50 * std_roi + 0.25 * min(worst, 0.0)

    return {
        "season_mean_roi": mean_roi,
        "season_std_roi": std_roi,
        "worst_season_roi": worst,
        "stability_score": float(score),
    }


def restrict_prior_window(prior: pd.DataFrame, current_week_key: int, window_weeks: int):
    if window_weeks >= 9999:
        return prior.copy()

    # week_key is season*100+week, so subtracting weeks numerically is unsafe
    # across season boundaries. Rank unique historical weeks instead.
    weeks = sorted(int(x) for x in prior["week_key"].unique())
    if len(weeks) <= window_weeks:
        return prior.copy()

    keep_weeks = set(weeks[-window_weeks:])
    return prior[prior["week_key"].isin(keep_weeks)].copy()


def candidate_metrics(
    prior: pd.DataFrame,
    lam: float,
    threshold: float,
    min_bets: int,
):
    x = prior[prior["win"].notna()].copy()
    if x.empty:
        return None

    x["p_stable"] = 0.5 + float(lam) * (
        x["calibrated_probability"] - 0.5
    )
    x["edge_stable"] = x["p_stable"] - x["break_even"]

    # Best spread side per game.
    x = (
        x.sort_values(
            ["game_id", "edge_stable"],
            ascending=[True, False],
        )
        .drop_duplicates(["game_id"])
    )

    bets = x[x["edge_stable"] >= float(threshold)].copy()
    if len(bets) < min_bets:
        return None

    roi = float(bets["profit"].mean())
    brier = brier_score(bets["p_stable"], bets["win"])
    lcb_roi = lower_confidence_roi(bets["profit"], z=1.0)
    stability = segment_stability_score(bets)

    # Composite score:
    # - prioritize conservative ROI
    # - reward stable season behavior
    # - modestly reward Brier below 0.25
    calibration_bonus = max(0.0, 0.25 - brier) if np.isfinite(brier) else 0.0

    score = (
        lcb_roi
        + 0.50 * stability["stability_score"]
        + 0.25 * calibration_bonus
    )

    return {
        "lambda": float(lam),
        "threshold": float(threshold),
        "bets": int(len(bets)),
        "roi": roi,
        "brier": brier,
        "lcb_roi": lcb_roi,
        **stability,
        "score": float(score),
    }


def choose_stable_config(
    prior: pd.DataFrame,
    current_week_key: int,
    lambdas,
    thresholds,
    windows_weeks,
    min_bets: int,
):
    candidates = []

    for window in windows_weeks:
        p = restrict_prior_window(prior, current_week_key, int(window))

        if len(p) < min_bets:
            continue

        for lam in lambdas:
            for threshold in thresholds:
                metrics = candidate_metrics(
                    p,
                    lam=float(lam),
                    threshold=float(threshold),
                    min_bets=min_bets,
                )
                if metrics is None:
                    continue
                metrics["window_weeks"] = int(window)
                metrics["prior_rows"] = int(len(p))
                candidates.append(metrics)

    if not candidates:
        return {
            "lambda": 0.25,
            "threshold": 0.02,
            "window_weeks": 52,
            "mode": "fallback",
            "score": np.nan,
            "prior_rows": int(len(prior)),
        }

    # Prefer the best score. Tie-break:
    # 1. lower Brier
    # 2. higher threshold (more conservative)
    # 3. shorter window (more recent)
    candidates.sort(
        key=lambda c: (
            c["score"],
            -c["brier"] if np.isfinite(c["brier"]) else -999.0,
            c["threshold"],
            -c["window_weeks"],
        ),
        reverse=True,
    )

    best = candidates[0].copy()
    best["mode"] = "tuned"
    return best


def apply_stability_filter(
    calibrated_sides: pd.DataFrame,
    lambdas=None,
    thresholds=None,
    windows_weeks=None,
    min_bets: int = 60,
):
    if lambdas is None:
        lambdas = DEFAULT_LAMBDAS
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if windows_weeks is None:
        windows_weeks = DEFAULT_WINDOWS_WEEKS

    d = calibrated_sides.copy()
    d = d[d["market"] == "spread"].copy()

    required = [
        "game_id", "season", "week", "week_key", "side",
        "calibrated_probability", "break_even", "profit", "result", "win",
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required calibrated-side columns: {missing}")

    d = d.sort_values(["week_key", "game_id", "side"]).reset_index(drop=True)

    outputs = []

    for week_key in sorted(int(x) for x in d["week_key"].unique()):
        current = d[d["week_key"] == week_key].copy()
        prior = d[d["week_key"] < week_key].copy()

        config = choose_stable_config(
            prior,
            current_week_key=week_key,
            lambdas=lambdas,
            thresholds=thresholds,
            windows_weeks=windows_weeks,
            min_bets=min_bets,
        )

        lam = float(config["lambda"])
        threshold = float(config["threshold"])

        current["stable_probability"] = 0.5 + lam * (
            current["calibrated_probability"] - 0.5
        )
        current["stable_edge"] = (
            current["stable_probability"] - current["break_even"]
        )

        # Exactly one spread side per game.
        current = (
            current.sort_values(
                ["game_id", "stable_edge"],
                ascending=[True, False],
            )
            .drop_duplicates(["game_id"])
        )

        current["stable_bet"] = current["stable_edge"] >= threshold
        current["selected_lambda"] = lam
        current["selected_threshold"] = threshold
        current["selected_window_weeks"] = int(config["window_weeks"])
        current["selection_mode"] = config["mode"]
        current["selection_score"] = config.get("score", np.nan)
        current["selection_prior_rows"] = int(config.get("prior_rows", len(prior)))

        outputs.append(current)

    if not outputs:
        return pd.DataFrame()

    return pd.concat(outputs, ignore_index=True)


def summarize(df: pd.DataFrame, bet_col: str, prob_col: str, edge_col: str):
    bets = df[df[bet_col]].copy()
    settled = bets[bets["result"] != "push"].copy()

    if bets.empty:
        return {
            "bets": 0,
            "settled": 0,
            "pushes": 0,
            "win_rate": None,
            "units": 0.0,
            "roi": None,
            "brier": None,
            "avg_edge": None,
        }

    units = float(bets["profit"].sum())

    return {
        "bets": int(len(bets)),
        "settled": int(len(settled)),
        "pushes": int((bets["result"] == "push").sum()),
        "win_rate": (
            float((settled["result"] == "win").mean())
            if len(settled) else None
        ),
        "units": units,
        "roi": float(units / len(bets)),
        "brier": (
            brier_score(settled[prob_col], settled["win"])
            if len(settled) else None
        ),
        "avg_edge": float(bets[edge_col].mean()),
    }


def build_v26_comparison(selected_rows: pd.DataFrame):
    s = selected_rows[selected_rows["market"] == "spread"].copy()

    # v2.6 already contains one best side per game/market.
    if "bet" not in s.columns:
        raise ValueError("v2.6 selected_rows.csv must contain 'bet' column.")

    return summarize(
        s,
        bet_col="bet",
        prob_col="selected_probability",
        edge_col="selected_edge",
    )


def season_summary(df: pd.DataFrame):
    rows = []
    for season in sorted(int(x) for x in df["season"].unique()):
        s = df[df["season"] == season].copy()
        summary = summarize(
            s,
            bet_col="stable_bet",
            prob_col="stable_probability",
            edge_col="stable_edge",
        )
        rows.append({"season": season, **summary})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--calibrated-sides",
        default="artifacts/calibration_v26/calibrated_sides.csv",
    )
    ap.add_argument(
        "--v26-selected-rows",
        default="artifacts/calibration_v26/selected_rows.csv",
    )
    ap.add_argument(
        "--output-dir",
        default="artifacts/spread_stability_v27",
    )
    ap.add_argument("--min-bets", type=int, default=60)
    args = ap.parse_args()

    calibrated = pd.read_csv(args.calibrated_sides)
    v26_selected = pd.read_csv(args.v26_selected_rows)

    stable = apply_stability_filter(
        calibrated,
        min_bets=args.min_bets,
    )

    v26_summary = build_v26_comparison(v26_selected)
    v27_summary = summarize(
        stable,
        bet_col="stable_bet",
        prob_col="stable_probability",
        edge_col="stable_edge",
    )

    season = season_summary(stable)

    report = {
        "validation": {
            "prior_weeks_only": True,
            "same_week_results_used": False,
            "rolling_window_tuning": True,
            "probability_shrinkage_tuned_on_prior_weeks_only": True,
            "threshold_tuned_on_prior_weeks_only": True,
            "season_stability_penalty_uses_prior_data_only": True,
        },
        "v26_spread_selector": v26_summary,
        "v27_spread_stability": v27_summary,
        "delta": {
            "roi_v27_minus_v26": (
                None if v26_summary["roi"] is None or v27_summary["roi"] is None
                else v27_summary["roi"] - v26_summary["roi"]
            ),
            "brier_v27_minus_v26": (
                None if v26_summary["brier"] is None or v27_summary["brier"] is None
                else v27_summary["brier"] - v26_summary["brier"]
            ),
            "win_rate_v27_minus_v26": (
                None if v26_summary["win_rate"] is None or v27_summary["win_rate"] is None
                else v27_summary["win_rate"] - v26_summary["win_rate"]
            ),
        },
        "candidate_grid": {
            "lambdas": DEFAULT_LAMBDAS,
            "thresholds": DEFAULT_THRESHOLDS,
            "rolling_windows_weeks": DEFAULT_WINDOWS_WEEKS,
            "min_bets": args.min_bets,
        },
    }

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    stable.to_csv(outdir / "stable_spread_rows.csv", index=False)
    season.to_csv(outdir / "summary_by_season.csv", index=False)
    (outdir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("\nSEASON SUMMARY")
    print(season.to_string(index=False))
    print(f"\nSaved outputs to {outdir}")


if __name__ == "__main__":
    main()
