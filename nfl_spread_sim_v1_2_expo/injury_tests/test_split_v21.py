from score_model.split_injury_oos_v21 import (
    QB_SPREAD_BASE,
    QB_TOTAL_BASE,
    SPREAD_PV_INJURY,
    TOTAL_BASIC_INJURY,
)

def test_split_feature_sets_are_distinct():
    assert "diff_pv_total_injury_impact" in SPREAD_PV_INJURY
    assert "sum_total_injury_impact" in TOTAL_BASIC_INJURY
    assert "diff_pv_total_injury_impact" not in TOTAL_BASIC_INJURY
    assert "sum_total_injury_impact" not in SPREAD_PV_INJURY

def test_qb_features_remain_in_both_markets():
    assert "diff_qb_epa_per_dropback" in QB_SPREAD_BASE
    assert "home_pre_qb_epa_per_dropback" in QB_TOTAL_BASE
