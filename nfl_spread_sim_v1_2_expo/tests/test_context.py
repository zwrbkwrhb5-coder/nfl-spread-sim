import pandas as pd
from src.context import add_schedule_context, haversine_miles

def test_haversine_positive():
    d = haversine_miles(39.0, -94.0, 42.7, -78.8)
    assert d > 500

def test_rest_days():
    games = pd.DataFrame([
        {"game_id":"g1","season":2025,"week":1,"game_date":"2025-09-01","home_team":"A","away_team":"B"},
        {"game_id":"g2","season":2025,"week":2,"game_date":"2025-09-08","home_team":"B","away_team":"A"},
    ])
    out = add_schedule_context(games)
    r = out[out["game_id"].eq("g2")].iloc[0]
    assert r["home_rest_days"] == 7
    assert r["away_rest_days"] == 7
