import pandas as pd
from src.college_prior import nfl_weight_from_dropbacks
from src.qb_injury_return import get_return_adjustment

def test_college_weight_fades():
    assert nfl_weight_from_dropbacks(0) == 0
    assert 0.49 < nfl_weight_from_dropbacks(500) < 0.51
    assert nfl_weight_from_dropbacks(5000) > 0.9

def test_return_prior_shrinks_small_sample():
    p = pd.DataFrame([{
        "injury_group": "ankle",
        "missed_bucket": "2_3",
        "sample_size": 4,
        "mean_epa_db_delta": -0.10,
        "mean_success_delta": -0.04,
        "mean_cpoe_delta": -2.0,
        "mean_sack_rate_delta": 0.01,
    }])
    r = get_return_adjustment(p, "ankle", 2)
    assert abs(r["qb_return_epa_adjustment"]) < 0.10
    assert r["return_prior_sample"] == 4
