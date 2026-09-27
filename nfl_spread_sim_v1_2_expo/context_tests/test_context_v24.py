import pandas as pd
from score_model.context_weather_v24 import normalize_context

def test_rest_and_weather_features():
    g = pd.DataFrame([{
        "game_id":"g1","season":2025,"week":5,
        "home_team":"AAA","away_team":"BBB",
        "home_rest":10,"away_rest":6,"div_game":True,
        "roof":"outdoors","surface":"grass","temp":30,"wind":18,
    }])
    x = normalize_context(g).iloc[0]
    assert x["rest_diff"] == 4
    assert x["away_short_week"] == 1
    assert x["home_long_rest"] == 1
    assert x["divisional_game"] == 1
    assert x["cold_game"] == 1
    assert x["windy_game"] == 1
