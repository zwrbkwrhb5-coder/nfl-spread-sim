import pandas as pd
from src.market import american_to_break_even, best_available_line

def test_minus_110_break_even():
    assert abs(american_to_break_even(-110) - 0.5238095) < 1e-6

def test_best_line_prefers_bigger_spread_number():
    df = pd.DataFrame([
        {"game_id":"g","snapshot_time":"2025-01-01T00:00:00Z","sportsbook":"A",
         "side_team":"BUF","spread":-3.5,"american_odds":-110,"market_phase":"current"},
        {"game_id":"g","snapshot_time":"2025-01-01T00:00:00Z","sportsbook":"B",
         "side_team":"BUF","spread":-3.0,"american_odds":-115,"market_phase":"current"},
    ])
    out = best_available_line(df, "g", "BUF")
    assert out.iloc[0]["spread"] == -3.0
