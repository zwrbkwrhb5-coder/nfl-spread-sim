from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def american_break_even(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def american_profit_per_unit(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    return 100.0 / (-odds)


def grade_binary_result(won: bool, pushed: bool, odds: float) -> float:
    if pushed:
        return 0.0
    return american_profit_per_unit(odds) if won else -1.0


def grade_spread(home_margin: float, home_spread: float) -> str:
    v = float(home_margin) + float(home_spread)
    if np.isclose(v, 0.0):
        return "push"
    return "home_cover" if v > 0 else "away_cover"


def grade_total(game_total: float, total_line: float) -> str:
    v = float(game_total) - float(total_line)
    if np.isclose(v, 0.0):
        return "push"
    return "over" if v > 0 else "under"


def empirical_probability_from_residuals(
    projection: float,
    residuals: np.ndarray,
    threshold: float,
    direction: str,
) -> tuple[float, float]:
    residuals = np.asarray(residuals, dtype=float)
    sims = projection + residuals
    if direction == "gt":
        pushes = np.isclose(sims, threshold)
        wins = sims > threshold
    elif direction == "lt":
        pushes = np.isclose(sims, threshold)
        wins = sims < threshold
    else:
        raise ValueError("direction must be gt or lt")

    push_prob = float(pushes.mean())
    win_prob = float(wins.mean())
    return win_prob, push_prob


@dataclass
class ProbabilityCalibrator:
    method: str = "isotonic"
    model: IsotonicRegression | None = None

    def fit(self, raw_prob: Iterable[float], outcome: Iterable[int]) -> "ProbabilityCalibrator":
        x = np.asarray(list(raw_prob), dtype=float)
        y = np.asarray(list(outcome), dtype=int)
        if len(np.unique(y)) < 2 or len(x) < 25:
            self.model = None
            return self
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.model.fit(x, y)
        return self

    def predict(self, raw_prob: Iterable[float]) -> np.ndarray:
        x = np.asarray(list(raw_prob), dtype=float)
        if self.model is None:
            return x
        return np.asarray(self.model.predict(x), dtype=float)


def make_edge_buckets(edge: pd.Series) -> pd.Series:
    bins = [-np.inf, 0, .01, .02, .03, .05, .075, .10, np.inf]
    labels = [
        "<=0%", "0-1%", "1-2%", "2-3%",
        "3-5%", "5-7.5%", "7.5-10%", "10%+"
    ]
    return pd.cut(edge, bins=bins, labels=labels, right=False)


def _safe_brier(df: pd.DataFrame, prob_col: str, outcome_col: str) -> float | None:
    x = df[[prob_col, outcome_col]].dropna()
    if x.empty:
        return None
    return float(brier_score_loss(x[outcome_col], x[prob_col]))


def summarize_market(df: pd.DataFrame, market: str) -> dict:
    d = df[df["market"].eq(market)].copy()
    if d.empty:
        return {"bets": 0}

    settled = d[~d["is_push"]].copy()
    return {
        "bets": int(len(d)),
        "settled_bets": int(len(settled)),
        "pushes": int(d["is_push"].sum()),
        "win_rate": float(settled["won"].mean()) if len(settled) else None,
        "units": float(d["profit_units"].sum()),
        "roi": float(d["profit_units"].sum() / len(settled)) if len(settled) else None,
        "avg_probability_edge": float(d["probability_edge"].mean()),
        "avg_line_edge_points": float(d["line_edge_points"].mean()),
        "brier_raw": _safe_brier(settled, "raw_probability", "won"),
        "brier_calibrated": _safe_brier(settled, "calibrated_probability", "won"),
    }


def edge_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["edge_bucket"] = make_edge_buckets(d["probability_edge"])
    rows = []
    for (market, bucket), g in d.groupby(["market", "edge_bucket"], observed=True):
        settled = g[~g["is_push"]]
        rows.append({
            "market": market,
            "edge_bucket": str(bucket),
            "bets": int(len(g)),
            "settled_bets": int(len(settled)),
            "win_rate": float(settled["won"].mean()) if len(settled) else np.nan,
            "units": float(g["profit_units"].sum()),
            "roi": float(g["profit_units"].sum() / len(settled)) if len(settled) else np.nan,
            "avg_probability_edge": float(g["probability_edge"].mean()),
            "avg_line_edge_points": float(g["line_edge_points"].mean()),
        })
    return pd.DataFrame(rows)


def closing_line_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "closing_line" not in df.columns:
        return pd.DataFrame()
    d = df[df["closing_line"].notna()].copy()
    if d.empty:
        return pd.DataFrame()

    rows = []
    for market, g in d.groupby("market"):
        rows.append({
            "market": market,
            "bets": int(len(g)),
            "avg_clv_points": float(g["clv_points"].mean()),
            "median_clv_points": float(g["clv_points"].median()),
            "positive_clv_rate": float((g["clv_points"] > 0).mean()),
        })
    return pd.DataFrame(rows)
