import pandas as pd
import numpy as np

from score_model.calibration_diagnostics_v32 import (
    add_walk_forward_platt,
    validate_time_safety,
    select_fixed_rule,
)


def sample_rows():
    rows = []
    for week in [202201, 202202, 202203]:
        for game_num in range(30):
            game_id = f"{week}_{game_num}"
            for side, p, win in [
                ("home", 0.60, 1.0 if game_num % 2 == 0 else 0.0),
                ("away", 0.40, 0.0 if game_num % 2 == 0 else 1.0),
            ]:
                rows.append({
                    "game_id": game_id,
                    "season": 2022,
                    "week": week % 100,
                    "week_key": week,
                    "market": "spread",
                    "side": side,
                    "raw_probability": p,
                    "odds": -110,
                    "break_even": 110 / 210,
                    "result": "win" if win else "loss",
                    "win": win,
                    "profit": 100 / 110 if win else -1.0,
                    "calibrated_probability": p,
                    "calibration_rows": 0 if week == 202201 else 60,
                })
    return pd.DataFrame(rows)


def test_platt_uses_only_prior_weeks():
    d = add_walk_forward_platt(sample_rows())
    audit = validate_time_safety(d)
    assert audit["platt_all_source_weeks_strictly_prior"] is True

    used = d[d["platt_mode"] == "platt"]
    assert (
        used["platt_max_source_week_key"]
        < used["week_key"]
    ).all()


def test_first_week_falls_back_to_raw():
    d = add_walk_forward_platt(sample_rows())
    first = d[d["week_key"] == 202201]
    assert (first["platt_mode"] == "raw_fallback").all()
    assert np.allclose(
        first["platt_probability"],
        first["raw_probability"],
    )


def test_fixed_selector_keeps_one_side():
    d = pd.DataFrame([
        {
            "game_id": "g1",
            "season": 2022,
            "week": 1,
            "week_key": 202201,
            "market": "spread",
            "side": "home",
            "odds": 100,
            "break_even": 0.5,
            "result": "win",
            "win": 1.0,
            "profit": 1.0,
            "raw_probability": 0.60,
            "calibration_rows": 10,
            "method": "raw",
            "probability": 0.60,
        },
        {
            "game_id": "g1",
            "season": 2022,
            "week": 1,
            "week_key": 202201,
            "market": "spread",
            "side": "away",
            "odds": 100,
            "break_even": 0.5,
            "result": "loss",
            "win": 0.0,
            "profit": -1.0,
            "raw_probability": 0.40,
            "calibration_rows": 10,
            "method": "raw",
            "probability": 0.40,
        },
    ])

    out = select_fixed_rule(d)
    assert len(out) == 1
    assert out.iloc[0]["side"] == "home"
