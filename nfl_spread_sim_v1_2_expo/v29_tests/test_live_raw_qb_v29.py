from score_model.live_raw_qb_v29 import (
    SPREAD_FEATURES,
    TOTAL_FEATURES,
    QB_METRICS,
)


def test_spread_uses_raw_qb_features():
    assert "diff_qb_epa_per_dropback" in SPREAD_FEATURES
    assert "diff_qb_epa_per_dropback_v28" not in SPREAD_FEATURES


def test_total_uses_raw_qb_features():
    assert "home_pre_qb_epa_per_dropback" in TOTAL_FEATURES
    assert "away_pre_qb_epa_per_dropback" in TOTAL_FEATURES
    assert "home_pre_qb_epa_per_dropback_v28" not in TOTAL_FEATURES


def test_total_keeps_context_weather():
    assert "divisional_game" in TOTAL_FEATURES
    assert "cold_degrees" in TOTAL_FEATURES
    assert "wind_over_10" in TOTAL_FEATURES


def test_spread_does_not_use_context_weather():
    assert "wind_over_10" not in SPREAD_FEATURES
    assert "cold_degrees" not in SPREAD_FEATURES


def test_qb_metric_count():
    assert len(QB_METRICS) == 6
