from __future__ import annotations
import pandas as pd

def build_top5_from_game_results(game_results: list[dict]) -> pd.DataFrame:
    """
    Combine each game's sportsbook opportunities and return the best 5
    slate-wide by probability edge.

    One game can technically contribute multiple rows; caller may later choose
    to restrict to one side per game if desired.
    """
    rows = []
    for g in game_results:
        for r in g.get("all_book_edges", []):
            rec = dict(r)
            rec["game_id"] = g["game_id"]
            rec["home_team"] = g["home_team"]
            rec["away_team"] = g["away_team"]
            rec["projected_home_margin"] = g["projected_home_margin"]
            rows.append(rec)

    if not rows:
        return pd.DataFrame()

    x = pd.DataFrame(rows)
    x = x.sort_values(
        ["probability_edge", "projected_differential"],
        ascending=[False, False],
    )

    # Keep the single best sportsbook opportunity for each game+side.
    x = x.drop_duplicates(["game_id", "side_team"], keep="first")

    return x.head(5).reset_index(drop=True)
