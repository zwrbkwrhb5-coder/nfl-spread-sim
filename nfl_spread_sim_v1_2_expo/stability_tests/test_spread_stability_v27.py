import pandas as pd

from score_model.spread_stability_v27 import (
    lower_confidence_roi,
    restrict_prior_window,
    segment_stability_score,
)


def test_lower_confidence_roi_penalizes_uncertainty():
    stable = [0.1] * 100
    volatile = [1.0, -0.8] * 50
    assert lower_confidence_roi(stable) > lower_confidence_roi(volatile)


def test_window_keeps_recent_unique_weeks():
    rows = []
    for i in range(10):
        rows.append({
            "week_key": 202200 + i + 1,
            "x": i,
        })
    d = pd.DataFrame(rows)
    out = restrict_prior_window(d, 202211, window_weeks=3)
    assert sorted(out["week_key"].unique()) == [202208, 202209, 202210]


def test_stability_penalizes_bad_worst_season():
    good = pd.DataFrame({
        "season": [2022]*20 + [2023]*20,
        "profit": [0.1]*40,
    })
    bad = pd.DataFrame({
        "season": [2022]*20 + [2023]*20,
        "profit": [0.5]*20 + [-0.3]*20,
    })
    assert (
        segment_stability_score(good)["stability_score"]
        > segment_stability_score(bad)["stability_score"]
    )
