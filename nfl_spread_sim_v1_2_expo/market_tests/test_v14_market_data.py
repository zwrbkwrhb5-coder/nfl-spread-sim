import pandas as pd
from score_model.fetch_nflverse_market import build_historical_market
from score_model.market_backtest_v14 import actual_spread_grade, actual_total_grade

def sample():
    return pd.DataFrame([{
        "game_id":"2024_01_A_B","season":2024,"game_type":"REG","week":1,
        "gameday":"2024-09-01","away_team":"A","away_score":20,
        "home_team":"B","home_score":27,"spread_line":3.5,
        "away_spread_odds":-105,"home_spread_odds":-115,
        "total_line":45.5,"under_odds":-108,"over_odds":-112,
    }])

def test_spread_sign_conversion():
    d=build_historical_market(sample(),2024,2024)
    assert d.iloc[0].home_spread == -3.5
    assert d.iloc[0].away_spread == 3.5

def test_prices_preserved():
    d=build_historical_market(sample(),2024,2024)
    assert d.iloc[0].home_spread_odds == -115
    assert d.iloc[0].over_odds == -112

def test_missing_price_imputed():
    s=sample()
    s.loc[0,"over_odds"]=None
    d=build_historical_market(s,2024,2024)
    assert d.iloc[0].over_odds == -110
    assert bool(d.iloc[0].over_odds_imputed)

def test_home_cover():
    assert actual_spread_grade(7,-3.5)=="home"

def test_away_cover():
    assert actual_spread_grade(2,-3.5)=="away"

def test_spread_push():
    assert actual_spread_grade(3,-3)=="push"

def test_over():
    assert actual_total_grade(48,45.5)=="over"

def test_under():
    assert actual_total_grade(42,45.5)=="under"

def test_total_push():
    assert actual_total_grade(45,45)=="push"
