import numpy as np
import pandas as pd

from score_model.qb_layer_v17 import (
    normalize_pbp,
    build_qb_game_table,
    add_pregame_qb_form,
)


def tiny_pbp():
    rows = []
    for game_id, season, week, team, qb, vals in [
        ("g1", 2022, 1, "AAA", "Q1", [0.2, 0.4, -0.1]),
        ("g2", 2022, 2, "AAA", "Q1", [0.5, 0.3, 0.1]),
        ("g3", 2022, 3, "AAA", "Q1", [0.8, 0.6, 0.2]),
    ]:
        for i, epa in enumerate(vals):
            rows.append({
                "game_id": game_id,
                "season": season,
                "week": week,
                "posteam": team,
                "passer_player_id": qb,
                "passer_player_name": "QB One",
                "pass_attempt": 1,
                "sack": 0,
                "qb_scramble": 0,
                "epa": epa,
                "success": float(epa > 0),
                "cpoe": 1.0 + i,
                "interception": 0,
                "yards_gained": 25 if i == 0 else 8,
            })
    return pd.DataFrame(rows)


def test_build_starter_qb_games():
    q = build_qb_game_table(tiny_pbp())
    assert len(q) == 3
    assert set(q["qb_id"]) == {"Q1"}
    assert all(q["dropbacks"] == 3)


def test_current_game_does_not_enter_pregame_form():
    q = build_qb_game_table(tiny_pbp())
    form = add_pregame_qb_form(q, halflife_games=2, min_games=1)

    g2 = form[form["game_id"] == "g2"].iloc[0]
    g1 = q[q["game_id"] == "g1"].iloc[0]

    assert np.isclose(
        g2["pre_qb_epa_per_dropback"],
        g1["qb_epa_per_dropback"],
    )


def test_prior_start_count():
    q = build_qb_game_table(tiny_pbp())
    form = add_pregame_qb_form(q, halflife_games=2, min_games=1)
    assert list(form["qb_prior_starts"]) == [0, 1, 2]
