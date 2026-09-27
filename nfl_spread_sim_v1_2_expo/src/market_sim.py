from __future__ import annotations
import numpy as np
import pandas as pd
from .sim_engine_v08 import simulate_spread_dynamic

def build_side_probability_function(
    projected_home_margin: float,
    residuals,
    target_sigma: float,
    home_team: str,
    away_team: str,
    n: int = 250_000,
    seed: int = 7,
    calibrator=None,
):
    """
    Returns function(side_team, spread) -> cover probability.

    We use a smaller default n for scanning many book lines.
    Final user-facing result can rerun the selected line at 1,000,000 sims.
    """
    def fn(side_team: str, spread: float) -> float:
        if side_team == home_team:
            home_spread = spread
            res = simulate_spread_dynamic(
                projected_home_margin=projected_home_margin,
                residuals=residuals,
                target_sigma=target_sigma,
                home_spread=home_spread,
                n=n,
                seed=seed,
                calibrator=calibrator,
            )
            return float(res["calibrated_cover_probability"])
        elif side_team == away_team:
            # Convert away-side spread to equivalent home spread.
            # Away +3.5 == Home -3.5.
            home_spread = -spread
            res = simulate_spread_dynamic(
                projected_home_margin=projected_home_margin,
                residuals=residuals,
                target_sigma=target_sigma,
                home_spread=home_spread,
                n=n,
                seed=seed,
                calibrator=calibrator,
            )
            # If away covers, home does not cover, ignoring pushes.
            return float(1.0 - res["calibrated_cover_probability"] - res["push_probability"])
        else:
            raise ValueError(f"Unknown side_team: {side_team}")

    return fn
