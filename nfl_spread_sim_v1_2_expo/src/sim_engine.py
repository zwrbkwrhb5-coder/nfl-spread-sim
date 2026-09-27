from __future__ import annotations
import numpy as np

def american_break_even(odds: int) -> float:
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)

def simulate_spread(
    projected_home_margin: float,
    residuals,
    home_spread: float,
    odds: int = -110,
    n: int = 1_000_000,
    seed: int = 7,
):
    """
    home_spread example:
      -3.5 means HOME is favored by 3.5
      +3.5 means HOME is underdog by 3.5

    A HOME spread covers when simulated_home_margin + home_spread > 0.
    """
    rng = np.random.default_rng(seed)
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size < 100:
        raise ValueError("Need at least 100 out-of-sample residuals for simulation.")

    noise = rng.choice(residuals, size=n, replace=True)
    margins = projected_home_margin + noise
    ats_value = margins + home_spread

    cover = float(np.mean(ats_value > 0))
    push = float(np.mean(np.isclose(ats_value, 0.0)))
    no_cover = 1.0 - cover - push
    breakeven = american_break_even(odds)

    return {
        "simulations": int(n),
        "projected_home_margin": float(projected_home_margin),
        "market_home_spread": float(home_spread),
        "american_odds": int(odds),
        "cover_probability": cover,
        "push_probability": push,
        "no_cover_probability": no_cover,
        "break_even_probability": breakeven,
        "probability_edge": cover - breakeven,
        "sim_mean_margin": float(np.mean(margins)),
        "sim_median_margin": float(np.median(margins)),
        "margin_p10": float(np.quantile(margins, 0.10)),
        "margin_p25": float(np.quantile(margins, 0.25)),
        "margin_p75": float(np.quantile(margins, 0.75)),
        "margin_p90": float(np.quantile(margins, 0.90)),
    }
