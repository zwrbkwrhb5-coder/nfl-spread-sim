import pandas as pd
from score_model.market_backtest_v16 import week_key, spread_grade, total_grade

def test_week_key():
    s=pd.Series([2022,2023])
    w=pd.Series([1,18])
    out=week_key(s,w)
    assert list(out)==[202201,202318]

def test_spread_grade():
    assert spread_grade(7,-3.5)=="home"
    assert spread_grade(2,-3.5)=="away"
    assert spread_grade(3,-3)=="push"

def test_total_grade():
    assert total_grade(48,45.5)=="over"
    assert total_grade(42,45.5)=="under"
    assert total_grade(45,45)=="push"
