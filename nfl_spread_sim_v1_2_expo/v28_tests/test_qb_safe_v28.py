import numpy as np
import pandas as pd

from score_model.qb_safe_v28 import (
    QB_METRICS,
    apply_past_only_qb_shrinkage,
    audit_no_future_targets,
)


def make_rows():
    rows = []
    for week in [1, 2, 3]:
        r = {
            "game_id": f"2022_{week:02d}_A_B",
            "season": 2022,
            "week": week,
            "home_qb_id": "H",
            "away_qb_id": "A",
            "home_qb_prior_dropbacks": (week - 1) * 40,
            "away_qb_prior_dropbacks": (week - 1) * 40,
        }
        for i, m in enumerate(QB_METRICS):
            r[f"home_pre_{m}"] = float(week + i)
            r[f"away_pre_{m}"] = float(week + i + 0.5)
        rows.append(r)
    return pd.DataFrame(rows)


def test_targets_are_strictly_prior_week():
    d = apply_past_only_qb_shrinkage(make_rows(), 100.0)
    audit = audit_no_future_targets(d)
    assert audit["past_only"]
    assert audit["future_or_same_week_target_rows"] == 0


def test_week_two_target_ignores_week_two_values():
    d = make_rows()
    out = apply_past_only_qb_shrinkage(d, 100.0)

    metric = QB_METRICS[0]
    wk2 = out[out["week"] == 2].iloc[0]

    # Week-2 target should be the pooled week-1 home/away mean only.
    expected = np.mean([
        d.loc[d["week"] == 1, f"home_pre_{metric}"].iloc[0],
        d.loc[d["week"] == 1, f"away_pre_{metric}"].iloc[0],
    ])
    assert abs(wk2[f"qb_target_{metric}_v28"] - expected) < 1e-12


def test_zero_sample_qb_shrinks_to_prior_target():
    d = make_rows()
    out = apply_past_only_qb_shrinkage(d, 100.0)

    metric = QB_METRICS[0]
    wk2 = out[out["week"] == 2].iloc[0]
    # In this synthetic data week 2 has 40 DB, so manually test formula.
    target = wk2[f"qb_target_{metric}_v28"]
    raw = wk2[f"home_pre_{metric}"]
    w = 40.0 / 140.0
    expected = target + w * (raw - target)
    assert abs(wk2[f"home_pre_{metric}_v28"] - expected) < 1e-12
