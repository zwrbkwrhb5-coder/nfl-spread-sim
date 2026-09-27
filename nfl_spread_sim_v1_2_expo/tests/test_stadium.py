import pandas as pd
from src.stadium import attach_stadium_environment

def test_stadium_attach():
    games = pd.DataFrame([{
        "game_id":"x", "season":2025, "home_team":"KC", "away_team":"BUF"
    }])
    stadiums = pd.DataFrame([{
        "team":"KC", "stadium":"Arrowhead Stadium",
        "season_from":2018, "season_to":2026,
        "capacity":76416, "loudness_index":97,
        "indoor":0, "altitude_ft":750,
    }])
    out, feats = attach_stadium_environment(games, stadiums)
    assert out.loc[0, "stadium_loudness_index"] == 97
    assert "stadium_loudness_per_1k" in feats
