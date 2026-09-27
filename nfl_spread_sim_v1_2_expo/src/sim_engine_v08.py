from __future__ import annotations
import numpy as np
from .sim_engine import american_break_even
from .uncertainty import scale_residuals_for_game

def simulate_spread_dynamic(
    projected_home_margin: float,
    residuals,
    target_sigma: float,
    home_spread: float,
    odds: int = -110,
    n: int = 1_000_000,
    seed: int = 7,
    calibrator=None,
):
    """
    Dynamic-volatility simulation.

    Steps:
    1. Rescale empirical OOS residuals to matchup-specific sigma.
    2. Bootstrap 1,000,000 margin errors.
    3. Calculate raw cover probability.
    4. Optionally calibrate that probability using historical cover outcomes.
    """
    rng = np.random.default_rng(seed)
    pool = scale_residuals_for_game(residuals, target_sigma)
    noise = rng.choice(pool, size=n, replace=True)

    margins = projected_home_margin + noise
    ats_value = margins + home_spread

    cover_raw = float(np.mean(ats_value > 0))
    push = float(np.mean(np.isclose(ats_value, 0.0)))
    no_cover_raw = max(0.0, 1.0 - cover_raw - push)

    cover_cal = cover_raw
    if calibrator is not None:
        cover_cal = float(calibrator.predict([cover_raw])[0])

    breakeven = american_break_even(odds)

    return {
        "simulations": int(n),
        "projected_home_margin": float(projected_home_margin),
        "market_home_spread": float(home_spread),
        "american_odds": int(odds),
        "target_sigma": float(target_sigma),
        "raw_cover_probability": cover_raw,
        "calibrated_cover_probability": cover_cal,
        "push_probability": push,
        "raw_no_cover_probability": no_cover_raw,
        "break_even_probability": breakeven,
        "raw_probability_edge": cover_raw - breakeven,
        "calibrated_probability_edge": cover_cal - breakeven,
        "sim_mean_margin": float(np.mean(margins)),
        "sim_median_margin": float(np.median(margins)),
        "margin_p05": float(np.quantile(margins, 0.05)),
        "margin_p10": float(np.quantile(margins, 0.10)),
        "margin_p25": float(np.quantile(margins, 0.25)),
        "margin_p75": float(np.quantile(margins, 0.75)),
        "margin_p90": float(np.quantile(margins, 0.90)),
        "margin_p95": float(np.quantile(margins, 0.95)),
    }
