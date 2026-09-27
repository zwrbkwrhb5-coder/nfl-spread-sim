from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

def build_uncertainty_targets(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Starting target for game-specific uncertainty:
    absolute out-of-sample margin residual.
    """
    x = preds.copy()
    x["abs_residual"] = x["residual"].abs()
    return x

def new_uncertainty_model(random_state: int = 17):
    """
    Predicts expected absolute error for each matchup.
    """
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=250,
        max_leaf_nodes=12,
        min_samples_leaf=25,
        l2_regularization=1.0,
        random_state=random_state,
    )

def fit_uncertainty_model(
    feature_frame: pd.DataFrame,
    walk_forward_preds: pd.DataFrame,
    features: list[str],
):
    """
    Join honest out-of-sample residuals back to matchup features and learn
    which games tend to have larger prediction errors.
    """
    t = build_uncertainty_targets(walk_forward_preds)
    joined = feature_frame.merge(
        t[["game_id","abs_residual"]],
        on="game_id",
        how="inner",
    ).dropna(subset=features + ["abs_residual"])

    if len(joined) < 200:
        raise ValueError("Need at least 200 historical OOS games for uncertainty model.")

    m = new_uncertainty_model()
    m.fit(joined[features], joined["abs_residual"])
    return m, joined

def expected_abs_to_sigma(expected_abs_error: float) -> float:
    """
    For a zero-mean normal variable:
      E|X| = sigma * sqrt(2/pi)

    We use this as a simple bridge from learned expected absolute error
    to a game-specific simulation scale.
    """
    return float(max(expected_abs_error, 0.5) / np.sqrt(2.0 / np.pi))

def scale_residuals_for_game(
    residuals,
    target_sigma: float,
):
    """
    Rescale empirical residuals to the target game volatility while preserving
    the historical residual shape/tails.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 100:
        raise ValueError("Need at least 100 historical residuals.")
    base_std = float(np.std(r, ddof=1))
    if base_std <= 0:
        raise ValueError("Residual standard deviation must be positive.")
    centered = r - np.mean(r)
    return centered * (target_sigma / base_std)

def fit_cover_calibrator(
    raw_cover_probs,
    actual_cover_results,
):
    """
    Isotonic calibration for spread-cover probabilities.

    actual_cover_results should be 0/1; pushes should generally be excluded
    or handled separately.
    """
    p = np.asarray(raw_cover_probs, dtype=float)
    y = np.asarray(actual_cover_results, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]

    if len(p) < 100:
        raise ValueError("Need at least 100 historical cover observations.")

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    return iso
