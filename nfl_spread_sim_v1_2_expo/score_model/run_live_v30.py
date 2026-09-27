from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd

from .live_inputs_v30 import harden_market, print_audit


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--market-csv",
        default="live_data/week3_2026_market.csv",
    )
    ap.add_argument(
        "--hardened-market",
        default="live_data/week3_2026_market_hardened_v30.csv",
    )
    ap.add_argument(
        "--training-csv",
        default="artifacts/dual_market/training_games_qb_context_v24.csv",
    )
    ap.add_argument(
        "--oos-csv",
        default="artifacts/dual_market/oos_qb_raw_context_v28.csv",
    )
    ap.add_argument(
        "--team-games",
        default="artifacts/team_games.parquet",
    )
    ap.add_argument(
        "--qb-games",
        default="artifacts/dual_market/qb_games_v17.csv",
    )
    ap.add_argument(
        "--calibrated-sides",
        default="artifacts/calibration_v28_raw/calibrated_sides.csv",
    )
    ap.add_argument(
        "--total-params",
        default="artifacts/calibration_v28_raw/latest_params.json",
    )
    ap.add_argument(
        "--spread-params",
        default="artifacts/spread_stability_v28_raw/latest_params.json",
    )
    ap.add_argument(
        "--output",
        default="artifacts/live/week3_2026_ranked_v30.csv",
    )
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=3)
    ap.add_argument("--market-max-age-minutes", type=float, default=30.0)
    ap.add_argument("--refresh-odds", action="store_true")
    ap.add_argument("--bookmaker", default=None)

    args = ap.parse_args()

    market = pd.read_csv(args.market_csv)

    hardened = harden_market(
        market,
        season=args.season,
        week=args.week,
        max_age_minutes=args.market_max_age_minutes,
        refresh_odds=args.refresh_odds,
        bookmaker=args.bookmaker,
        api_key=os.environ.get("THE_ODDS_API_KEY"),
    )

    print_audit(hardened)

    hp = Path(args.hardened_market)
    hp.parent.mkdir(parents=True, exist_ok=True)

    # v2.9 can safely keep spread-ready games even if weather is unavailable;
    # its own totals gate will reject totals lacking temp/wind.
    runnable = hardened[
        hardened["spread_input_ready_v30"]
    ].copy()

    if runnable.empty:
        hardened.to_csv(hp, index=False)
        raise SystemExit(
            "\nNo games passed v3.0 input hardening. "
            "Do not loosen the model threshold. Fix the input warnings above."
        )

    runnable.to_csv(hp, index=False)

    print(
        f"\nRunning v2.9 model core on {len(runnable)} "
        f"v3.0-approved games..."
    )

    cmd = [
        "python",
        "-m",
        "score_model.live_raw_qb_v29",
        "--training-csv",
        args.training_csv,
        "--oos-csv",
        args.oos_csv,
        "--team-games",
        args.team_games,
        "--qb-games",
        args.qb_games,
        "--market-csv",
        str(hp),
        "--calibrated-sides",
        args.calibrated_sides,
        "--total-params",
        args.total_params,
        "--spread-params",
        args.spread_params,
        "--season",
        str(args.season),
        "--week",
        str(args.week),
        "--output",
        args.output,
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
