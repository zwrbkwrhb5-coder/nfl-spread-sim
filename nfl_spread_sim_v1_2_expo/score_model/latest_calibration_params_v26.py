from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .calibration_selection_v26 import (
    choose_lambda_and_threshold,
    DEFAULT_THRESHOLDS,
    DEFAULT_LAMBDAS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selected-rows",
        default="artifacts/calibration_v26/selected_rows.csv",
    )
    ap.add_argument(
        "--output",
        default="artifacts/calibration_v26/latest_params.json",
    )
    ap.add_argument("--min-bets-for-threshold", type=int, default=75)
    args = ap.parse_args()

    d = pd.read_csv(args.selected_rows)

    # Use the complete historical time-safe rows as the prior sample for the
    # next live slate. No live result is included.
    report = {}

    for market_name in ["spread", "total"]:
        prior = d[d["market"] == market_name].copy()

        lam, threshold, mode = choose_lambda_and_threshold(
            prior,
            DEFAULT_THRESHOLDS,
            DEFAULT_LAMBDAS,
            min_bets_for_threshold=args.min_bets_for_threshold,
        )

        report[market_name] = {
            "probability_shrinkage_lambda": lam,
            "minimum_edge": threshold,
            "selection_mode": mode,
            "prior_rows": int(len(prior)),
        }

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
