from __future__ import annotations
import numpy as np
import pandas as pd

STADIUM_FEATURES = [
    "stadium_loudness_index",
    "stadium_loudness_per_1k",
    "stadium_capacity_k",
    "stadium_indoor",
    "stadium_altitude_kft",
]

def validate_stadiums(stadiums: pd.DataFrame) -> pd.DataFrame:
    required = [
        "team", "stadium", "season_from", "season_to",
        "capacity", "loudness_index", "indoor", "altitude_ft",
    ]
    missing = [c for c in required if c not in stadiums.columns]
    if missing:
        raise ValueError(f"Missing stadium columns: {missing}")
    s = stadiums.copy()
    s["capacity"] = pd.to_numeric(s["capacity"], errors="coerce")
    s["loudness_index"] = pd.to_numeric(s["loudness_index"], errors="coerce")
    s["indoor"] = pd.to_numeric(s["indoor"], errors="coerce")
    s["altitude_ft"] = pd.to_numeric(s["altitude_ft"], errors="coerce")
    return s

def attach_stadium_environment(games: pd.DataFrame, stadiums: pd.DataFrame):
    """
    Attach season-appropriate home-stadium attributes.

    The stadium table is versioned by season so relocations/renames/capacity
    changes can be represented without leaking future information.
    """
    s = validate_stadiums(stadiums)
    g = games.copy()
    rows = []
    for _, game in g.iterrows():
        hit = s[
            s["team"].eq(game["home_team"]) &
            s["season_from"].le(game["season"]) &
            s["season_to"].ge(game["season"])
        ]
        rec = game.to_dict()
        if hit.empty:
            rec.update({
                "stadium_name": np.nan,
                "stadium_loudness_index": np.nan,
                "stadium_loudness_per_1k": np.nan,
                "stadium_capacity_k": np.nan,
                "stadium_indoor": np.nan,
                "stadium_altitude_kft": np.nan,
            })
        else:
            h = hit.iloc[-1]
            cap = float(h["capacity"])
            loud = float(h["loudness_index"])
            rec.update({
                "stadium_name": h["stadium"],
                "stadium_loudness_index": loud,
                "stadium_loudness_per_1k": loud / (cap / 1000.0) if cap else np.nan,
                "stadium_capacity_k": cap / 1000.0,
                "stadium_indoor": float(h["indoor"]),
                "stadium_altitude_kft": float(h["altitude_ft"]) / 1000.0,
            })
        rows.append(rec)
    return pd.DataFrame(rows), STADIUM_FEATURES.copy()

def add_noise_interactions(games: pd.DataFrame):
    """
    Interactions let the model learn that the same crowd may affect different
    visiting offenses differently.

    Requires QB differential fields when available. These are deliberately
    simple starting interactions and must earn their place in backtesting.
    """
    g = games.copy()
    feats = []

    # Higher away sack rate should plausibly increase vulnerability to noise.
    if "away_pre_qb_sack_rate" in g.columns:
        name = "noise_x_away_qb_sack_rate"
        g[name] = g["stadium_loudness_index"] * g["away_pre_qb_sack_rate"]
        feats.append(name)

    # Negative away CPOE can proxy passing instability; preserve direction.
    if "away_pre_qb_cpoe" in g.columns:
        name = "noise_x_away_qb_cpoe"
        g[name] = g["stadium_loudness_index"] * g["away_pre_qb_cpoe"]
        feats.append(name)

    name = "noise_x_indoor"
    g[name] = g["stadium_loudness_index"] * g["stadium_indoor"]
    feats.append(name)

    return g, feats
