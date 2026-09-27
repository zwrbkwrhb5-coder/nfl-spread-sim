import pandas as pd
from score_model.injury_layer_v19 import normalize_injuries, aggregate_team_week

def test_qb_out_has_larger_weight_than_rb_questionable():
    raw=pd.DataFrame([
        {"season":2024,"week":1,"team":"AAA","position":"QB","status":"Out","player_id":"q1"},
        {"season":2024,"week":1,"team":"AAA","position":"RB","status":"Questionable","player_id":"r1"},
    ])
    n=normalize_injuries(raw)
    qb=n[n.position=="QB"].iloc[0].injury_impact
    rb=n[n.position=="RB"].iloc[0].injury_impact
    assert qb > rb

def test_team_week_aggregation():
    raw=pd.DataFrame([
        {"season":2024,"week":1,"team":"AAA","position":"QB","status":"Out","player_id":"q1"},
        {"season":2024,"week":1,"team":"AAA","position":"CB","status":"Questionable","player_id":"c1"},
    ])
    n=normalize_injuries(raw)
    g=aggregate_team_week(n)
    assert len(g)==1
    assert g.iloc[0].qb_injury_impact > 0
    assert g.iloc[0].secondary_injury_impact > 0
