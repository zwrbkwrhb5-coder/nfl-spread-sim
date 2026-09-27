import pandas as pd
from src.features import merge_qb_features_into_matchups

def test_qb_feature_difference():
    games = pd.DataFrame([{
        "game_id": "g1",
        "home_team": "BUF",
        "away_team": "MIA",
        "season": 2025,
        "week": 1,
        "home_score": 30,
        "away_score": 20,
        "home_margin": 10,
    }])
    q = pd.DataFrame([
        {
            "game_id": "g1", "team": "BUF", "player_id": "h",
            "player_name": "Home", "starter": 1,
            "pre_qb_epa_per_dropback": 0.2,
        },
        {
            "game_id": "g1", "team": "MIA", "player_id": "a",
            "player_name": "Away", "starter": 1,
            "pre_qb_epa_per_dropback": 0.1,
        },
    ])
    out, feats = merge_qb_features_into_matchups(games, q)
    assert "diff_qb_epa_per_dropback" in feats
    assert abs(out.loc[0, "diff_qb_epa_per_dropback"] - 0.1) < 1e-9
