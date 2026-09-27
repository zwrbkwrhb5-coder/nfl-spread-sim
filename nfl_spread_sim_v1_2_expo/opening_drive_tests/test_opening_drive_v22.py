import pandas as pd
from score_model.opening_drive_v22 import add_pregame_tendencies

def test_current_game_not_used_in_pregame_tendency():
    d = pd.DataFrame([
        {"game_id":"g1","season":2022,"week":1,"team":"AAA","opponent":"BBB",
         "opening_points":7.0,"opening_epa":2.0,"opening_success_rate":.6,
         "opening_turnover":0.0,"opening_scored":1.0,"opening_td":1.0,"opening_fg":0.0},
        {"game_id":"g2","season":2022,"week":2,"team":"AAA","opponent":"CCC",
         "opening_points":0.0,"opening_epa":-1.0,"opening_success_rate":.2,
         "opening_turnover":1.0,"opening_scored":0.0,"opening_td":0.0,"opening_fg":0.0},
    ])
    out = add_pregame_tendencies(d, halflife_games=2, min_games=1)
    g2 = out[out.game_id=="g2"].iloc[0]
    assert abs(g2["pre_off_opening_points"] - 7.0) < 1e-9
