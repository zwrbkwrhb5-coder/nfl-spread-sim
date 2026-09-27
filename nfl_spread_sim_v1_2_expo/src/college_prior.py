from __future__ import annotations
import numpy as np
import pandas as pd

def nfl_weight_from_dropbacks(
    nfl_dropbacks: int,
    transition_scale: int = 500,
) -> float:
    """
    Smoothly shifts weight from college prior to NFL evidence.

    0 NFL dropbacks -> 0% NFL / 100% college
    transition_scale dropbacks -> 50% NFL / 50% college
    many NFL dropbacks -> approaches 100% NFL

    This is a starting curve; backtesting should tune it.
    """
    nfl_dropbacks = max(int(nfl_dropbacks), 0)
    return nfl_dropbacks / (nfl_dropbacks + transition_scale)

def blend_college_nfl_metric(
    college_value: float,
    nfl_value: float,
    nfl_dropbacks: int,
    transition_scale: int = 500,
) -> float:
    w = nfl_weight_from_dropbacks(nfl_dropbacks, transition_scale)
    if pd.isna(college_value):
        return float(nfl_value)
    if pd.isna(nfl_value):
        return float(college_value)
    return float((1 - w) * college_value + w * nfl_value)

def build_qb_prior_features(
    college_qb: pd.DataFrame,
    nfl_qb_state: pd.DataFrame,
    transition_scale: int = 500,
) -> pd.DataFrame:
    """
    Expected columns in college_qb:
      player_id, college_epa_db, college_cpoe, college_sack_rate,
      college_explosive_pass_rate, college_opponent_strength

    Expected columns in nfl_qb_state:
      player_id, nfl_dropbacks, qb_epa_per_dropback, qb_cpoe,
      qb_sack_rate, qb_explosive_pass_rate
    """
    x = nfl_qb_state.merge(college_qb, on="player_id", how="left")

    x["blended_qb_epa_db"] = x.apply(
        lambda r: blend_college_nfl_metric(
            r.get("college_epa_db", np.nan),
            r.get("qb_epa_per_dropback", np.nan),
            r.get("nfl_dropbacks", 0),
            transition_scale,
        ),
        axis=1,
    )
    x["blended_qb_cpoe"] = x.apply(
        lambda r: blend_college_nfl_metric(
            r.get("college_cpoe", np.nan),
            r.get("qb_cpoe", np.nan),
            r.get("nfl_dropbacks", 0),
            transition_scale,
        ),
        axis=1,
    )
    return x
