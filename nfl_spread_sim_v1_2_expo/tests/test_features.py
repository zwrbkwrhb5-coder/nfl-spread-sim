import pandas as pd
from src.features import add_pregame_recency_features

def test_pregame_feature_is_shifted():
    df = pd.DataFrame({
        "team": ["BUF"] * 4,
        "season": [2025] * 4,
        "week": [1,2,3,4],
        "game_id": ["g1","g2","g3","g4"],
        "off_epa": [1.0, 2.0, 3.0, 100.0],
        "def_epa": [0.0]*4,
        "off_success": [0.0]*4,
        "def_success": [0.0]*4,
        "explosive_pass_rate": [0.0]*4,
        "sack_allowed_rate": [0.0]*4,
        "def_sack_rate": [0.0]*4,
        "turnover_rate": [0.0]*4,
        "points_for": [0.0]*4,
        "points_against": [0.0]*4,
    })
    out = add_pregame_recency_features(df, halflife_games=2, min_games=1)
    # Week 4's pregame feature must NOT include the current value 100.
    assert out.loc[3, "pre_off_epa"] < 10.0
