from score_model.live_consistent_v28 import (
    SPREAD_FEATURES,
    TOTAL_FEATURES,
)


def test_safe_qb_features_are_used_live():
    assert "diff_qb_epa_per_dropback_v28" in SPREAD_FEATURES
    assert "home_pre_qb_epa_per_dropback_v28" in TOTAL_FEATURES
    assert "away_pre_qb_epa_per_dropback_v28" in TOTAL_FEATURES


def test_totals_use_real_adjusted_features():
    assert "home_pre_adj_off_epa" in TOTAL_FEATURES
    assert "away_pre_adj_def_epa" in TOTAL_FEATURES


def test_context_remains_totals_only():
    assert "wind_over_10" in TOTAL_FEATURES
    assert "wind_over_10" not in SPREAD_FEATURES
