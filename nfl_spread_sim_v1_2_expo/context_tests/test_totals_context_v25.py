from score_model.totals_context_oos_v25 import (
    SPREAD_FEATURES,
    TOTAL_BASE,
    TOTAL_CONTEXT,
)


def test_context_only_changes_totals_feature_set():
    assert "divisional_game" in TOTAL_CONTEXT
    assert "wind_over_10" in TOTAL_CONTEXT
    assert "cold_degrees" in TOTAL_CONTEXT

    assert "divisional_game" not in SPREAD_FEATURES
    assert "wind_over_10" not in SPREAD_FEATURES
    assert "cold_degrees" not in SPREAD_FEATURES


def test_qb_remains_in_both_models():
    assert "diff_qb_epa_per_dropback" in SPREAD_FEATURES
    assert "home_pre_qb_epa_per_dropback" in TOTAL_BASE
    assert "away_pre_qb_epa_per_dropback" in TOTAL_BASE
