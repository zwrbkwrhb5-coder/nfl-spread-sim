from __future__ import annotations
import numpy as np
import pandas as pd
from .predict import matchup_vector as team_matchup_vector
from .qb import QB_METRICS

def _current_qb_metric(rows, col, halflife, min_games):
    s = rows[col].dropna()
    if len(s) < min_games:
        return np.nan
    return float(
        s.ewm(
            halflife=halflife,
            adjust=False,
            min_periods=min_games,
        ).mean().iloc[-1]
    )

def current_qb_state(qb_games, player_id, halflife=5.0, min_games=2):
    rows = qb_games[qb_games["player_id"].eq(player_id)].sort_values(
        ["season", "week", "game_id"]
    )
    if rows.empty:
        raise ValueError(f"No QB history for {player_id}")

    return {
        col: _current_qb_metric(rows, col, halflife, min_games)
        for col in QB_METRICS
    }

def build_future_matchup_vector(
    team_games,
    qb_games,
    home,
    away,
    home_qb_id,
    away_qb_id,
    all_features,
    team_halflife=6.0,
    qb_halflife=5.0,
    min_team_games=3,
    min_qb_games=2,
):
    team_features = [f for f in all_features if not f.startswith("diff_qb_")]
    qb_features = [f for f in all_features if f.startswith("diff_qb_")]

    X_team = team_matchup_vector(
        team_games,
        home,
        away,
        team_features,
        halflife_games=team_halflife,
        min_games=min_team_games,
    )

    h = current_qb_state(qb_games, home_qb_id, qb_halflife, min_qb_games)
    a = current_qb_state(qb_games, away_qb_id, qb_halflife, min_qb_games)

    row = X_team.iloc[0].to_dict()
    for f in qb_features:
        base = f.removeprefix("diff_")
        row[f] = h[base] - a[base]

    return pd.DataFrame([row], columns=all_features)
