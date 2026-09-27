from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .spread_stability_v27 import (
    choose_stable_config,
    DEFAULT_LAMBDAS,
    DEFAULT_THRESHOLDS,
    DEFAULT_WINDOWS_WEEKS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--calibrated-sides",
        default="artifacts/calibration_v26/calibrated_sides.csv",
    )
    ap.add_argument(
        "--output",
        default="artifacts/spread_stability_v27/latest_params.json",
    )
    ap.add_argument("--min-bets", type=int, default=60)
    args = ap.parse_args()

    d = pd.read_csv(args.calibrated_sides)
    d = d[d["market"] == "spread"].copy()

    next_week_key = int(d["week_key"].max()) + 1

    config = choose_stable_config(
        d,
        current_week_key=next_week_key,
        lambdas=DEFAULT_LAMBDAS,
        thresholds=DEFAULT_THRESHOLDS,
        windows_weeks=DEFAULT_WINDOWS_WEEKS,
        min_bets=args.min_bets,
    )

    report = {
        "spread": {
            "probability_shrinkage_lambda": float(config["lambda"]),
            "minimum_edge": float(config["threshold"]),
            "rolling_window_weeks": int(config["window_weeks"]),
            "selection_mode": config["mode"],
            "selection_score": (
                None if pd.isna(config.get("score"))
                else float(config["score"])
            ),
            "prior_rows": int(config.get("prior_rows", len(d))),
        }
    }

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
