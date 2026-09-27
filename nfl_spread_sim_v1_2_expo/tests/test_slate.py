from src.slate import build_top5_from_game_results

def test_top5_sorting():
    results = [
        {
            "game_id":"g1","home_team":"A","away_team":"B","projected_home_margin":3.0,
            "all_book_edges":[
                {"side_team":"A","sportsbook":"X","probability_edge":0.08,"projected_differential":2.0},
                {"side_team":"A","sportsbook":"Y","probability_edge":0.06,"projected_differential":2.5},
            ],
        },
        {
            "game_id":"g2","home_team":"C","away_team":"D","projected_home_margin":1.0,
            "all_book_edges":[
                {"side_team":"D","sportsbook":"Z","probability_edge":0.09,"projected_differential":1.5}
            ],
        },
    ]
    top = build_top5_from_game_results(results)
    assert top.iloc[0]["game_id"] == "g2"
    assert len(top) == 2
