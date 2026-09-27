from __future__ import annotations
import numpy as np
import pandas as pd
from .features import BASE_COLS, ADJ_COLS

def _ewm_current(values: pd.Series, halflife_games: float, min_games: int) -> float:
    s = values.dropna()
    if len(s) < min_games:
        raise ValueError(f"Need at least {min_games} valid games; found {len(s)}")
    return float(
        s.ewm(halflife=halflife_games, adjust=False, min_periods=min_games)
         .mean()
         .iloc[-1]
    )

def current_team_state(
    team_games: pd.DataFrame,
    team: str,
    halflife_games: float = 6.0,
    min_games: int = 3,
) -> dict:
    """
    Future-game state: all completed games are available, so use the latest
    exponentially weighted value INCLUDING the most recent completed game.
    """
    rows = (
        team_games[team_games["team"].eq(team)]
        .sort_values(["season", "week", "game_id"])
        .copy()
    )
    if rows.empty:
        raise ValueError(f"No rows found for team {team}")

    state = {}
    for col in BASE_COLS + ADJ_COLS:
        if col not in rows:
            raise KeyError(f"Missing expected column: {col}")
        state[col] = _ewm_current(rows[col], halflife_games, min_games)
    return state

def matchup_vector(
    team_games: pd.DataFrame,
    home: str,
    away: str,
    features: list[str],
    halflife_games: float = 6.0,
    min_games: int = 3,
):
    h = current_team_state(team_games, home, halflife_games, min_games)
    a = current_team_state(team_games, away, halflife_games, min_games)

    row = {}
    for f in features:
        base = f.removeprefix("diff_")
        row[f] = h[base] - a[base]

    return pd.DataFrame([row], columns=features)
