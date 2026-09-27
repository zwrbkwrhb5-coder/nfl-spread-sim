import pandas as pd

from score_model.multi_book_live_v31 import (
    ODDS_API_TO_INDIANA,
    add_quote_freshness,
    select_best_per_game_market,
)


def test_key_aliases_for_indiana_books():
    assert ODDS_API_TO_INDIANA["williamhill_us"] == "caesars"
    assert ODDS_API_TO_INDIANA["espnbet"] == "thescore"
    assert ODDS_API_TO_INDIANA["hardrockbet"] == "hardrockbet"


def test_quote_freshness_requires_recent_time():
    now = pd.Timestamp.now(tz="UTC")
    d = pd.DataFrame({
        "quote_updated_at": [
            now.isoformat(),
            (now - pd.Timedelta(hours=2)).isoformat(),
        ]
    })
    out = add_quote_freshness(d, 30)
    assert bool(out.iloc[0]["quote_fresh"])
    assert not bool(out.iloc[1]["quote_fresh"])


def test_best_selection_is_one_per_game_market():
    now = pd.Timestamp.now(tz="UTC")
    d = pd.DataFrame([
        {
            "game_id": "g1",
            "market": "spread",
            "edge": 0.02,
            "qualifies": True,
            "quote_fresh": True,
            "quote_updated_at": now,
        },
        {
            "game_id": "g1",
            "market": "spread",
            "edge": 0.03,
            "qualifies": True,
            "quote_fresh": True,
            "quote_updated_at": now,
        },
        {
            "game_id": "g1",
            "market": "total",
            "edge": 0.01,
            "qualifies": True,
            "quote_fresh": True,
            "quote_updated_at": now,
        },
    ])
    out = select_best_per_game_market(d)
    assert len(out) == 2
    spread = out[out["market"] == "spread"].iloc[0]
    assert spread["edge"] == 0.03
