import pandas as pd
from src.opening_drive import add_opening_drive_pregame_form

def test_opening_drive_shifted():
    df = pd.DataFrame({
        "team":["A"]*4,
        "season":[2025]*4,
        "week":[1,2,3,4],
        "game_id":["g1","g2","g3","g4"],
        "od_points":[0,7,3,100],
        "od_scored":[0,1,1,1],
        "od_td":[0,1,0,1],
        "od_fg":[0,0,1,0],
        "od_turnover":[0,0,0,0],
        "od_epa":[-1,2,1,50],
    })
    out = add_opening_drive_pregame_form(df, halflife_games=2, min_games=1)
    assert out.loc[3, "pre_od_points"] < 20
