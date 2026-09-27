import pandas as pd

from score_model.live_inputs_v30 import (
    normalize_team,
    parse_depth_chart_qbs,
    add_market_freshness,
    roof_classification,
)


def test_team_aliases():
    assert normalize_team("LAR") == "LA"
    assert normalize_team("JAC") == "JAX"
    assert normalize_team("WSH") == "WAS"


def test_depth_chart_latest_qb1():
    d = pd.DataFrame([
        {
            "dt": "2026-09-20T10:00:00Z",
            "team": "DAL",
            "player_name": "QB Old",
            "gsis_id": "old",
            "pos_abb": "QB",
            "pos_rank": 1,
        },
        {
            "dt": "2026-09-26T10:00:00Z",
            "team": "DAL",
            "player_name": "QB One",
            "gsis_id": "one",
            "pos_abb": "QB",
            "pos_rank": 1,
        },
        {
            "dt": "2026-09-26T10:00:00Z",
            "team": "DAL",
            "player_name": "QB Two",
            "gsis_id": "two",
            "pos_abb": "QB",
            "pos_rank": 2,
        },
    ])

    q = parse_depth_chart_qbs(d)
    r = q[q["team"] == "DAL"].iloc[0]
    assert r["qb_id"] == "one"
    assert r["qb_name"] == "QB One"


def test_freshness_requires_timestamp():
    d = pd.DataFrame([{"market_updated_at": None}])
    out = add_market_freshness(d, 30)
    assert not bool(out.iloc[0]["market_fresh_v30"])


def test_recent_market_is_fresh():
    now = pd.Timestamp.now(tz="UTC").isoformat()
    d = pd.DataFrame([{"market_updated_at": now}])
    out = add_market_freshness(d, 30)
    assert bool(out.iloc[0]["market_fresh_v30"])


def test_retractable_roof_without_status_is_unresolved():
    r = pd.Series({
        "home_team": "DAL",
        "roof": None,
        "roof_override": None,
    })
    kind, _, known = roof_classification(r)
    assert kind == "unknown_retractable"
    assert not known
