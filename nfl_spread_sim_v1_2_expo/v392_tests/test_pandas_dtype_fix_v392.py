import pandas as pd

from score_model.forward_test_v39 import normalize_ledger_dtypes


def test_blank_text_columns_can_accept_strings():
    d = pd.DataFrame({
        "closing_snapshot_label": [float("nan")],
        "closing_captured_utc": [float("nan")],
        "close_best_book": [float("nan")],
        "result": [float("nan")],
    })

    d = normalize_ledger_dtypes(d)

    d.loc[0, "closing_snapshot_label"] = "closing_candidate"
    d.loc[0, "closing_captured_utc"] = "2026-09-27T17:00:00+00:00"
    d.loc[0, "close_best_book"] = "FanDuel"
    d.loc[0, "result"] = "win"

    assert d.loc[0, "closing_snapshot_label"] == "closing_candidate"
    assert d.loc[0, "close_best_book"] == "FanDuel"
    assert d.loc[0, "result"] == "win"
