import pandas as pd
from src.injuries import (
    validate_player_availability,
    build_team_game_injury_features,
)

def test_qb_out_has_more_impact_than_rb_questionable():
    df = pd.DataFrame([
        {
            "game_id":"g","season":2025,"week":1,"team":"A",
            "player_id":"q","player_name":"Q","position":"QB","status":"out",
            "snap_share_prior":1.0,"starter_flag":1,"player_value_prior":1.0,
        },
        {
            "game_id":"g","season":2025,"week":1,"team":"A",
            "player_id":"r","player_name":"R","position":"RB","status":"questionable",
            "snap_share_prior":0.5,"starter_flag":0,"player_value_prior":0.4,
        },
    ])
    x = validate_player_availability(df)
    out = build_team_game_injury_features(x)
    assert out.loc[0, "injury_qb_impact"] > out.loc[0, "injury_rb_impact"]
