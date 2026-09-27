import pandas as pd

from score_model.historical_market_timing_v38 import audit_market
from score_model.live_totals_shadow_v38 import (
    total_consensus,
)


def test_closing_alias_classification():
    d = pd.DataFrame({
        "home_spread": [-3.0, 2.5],
        "closing_home_spread": [-3.0, 2.5],
        "total_line": [44.5, 47.0],
        "closing_total": [44.5, 47.0],
    })

    _, classification = audit_market(d)
    assert classification == "closing_alias"


def test_consensus_collapses_sides_per_book():
    d = pd.DataFrame([
        {
            "game_id": "g1",
            "away_team": "A",
            "home_team": "B",
            "book": "Book1",
            "market": "total",
            "side": "over",
            "line": 44.5,
            "odds": -110,
            "model_total": 46.0,
            "quote_fresh": True,
            "quote_updated_at": "2026-09-27T16:00:00Z",
        },
        {
            "game_id": "g1",
            "away_team": "A",
            "home_team": "B",
            "book": "Book1",
            "market": "total",
            "side": "under",
            "line": 44.5,
            "odds": -110,
            "model_total": 46.0,
            "quote_fresh": True,
            "quote_updated_at": "2026-09-27T16:00:00Z",
        },
        {
            "game_id": "g1",
            "away_team": "A",
            "home_team": "B",
            "book": "Book2",
            "market": "total",
            "side": "over",
            "line": 45.5,
            "odds": -110,
            "model_total": 46.0,
            "quote_fresh": True,
            "quote_updated_at": "2026-09-27T16:00:00Z",
        },
        {
            "game_id": "g1",
            "away_team": "A",
            "home_team": "B",
            "book": "Book2",
            "market": "total",
            "side": "under",
            "line": 45.5,
            "odds": -110,
            "model_total": 46.0,
            "quote_fresh": True,
            "quote_updated_at": "2026-09-27T16:00:00Z",
        },
    ])

    out = total_consensus(d)
    r = out.iloc[0]

    assert r["books_in_consensus"] == 2
    assert r["current_consensus_total"] == 45.0
