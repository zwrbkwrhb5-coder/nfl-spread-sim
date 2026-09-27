from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

from .market import (
    validate_market_snapshot,
    build_consensus_line,
    build_line_movement,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="data/market_odds_template.csv")
    args = ap.parse_args()

    market = pd.read_csv(args.market)
    market = validate_market_snapshot(market)

    current = build_consensus_line(market, "current")
    movement = build_line_movement(market)

    out = Path("artifacts")
    out.mkdir(exist_ok=True)

    current.to_csv(out/"market_current_consensus_v09.csv", index=False)
    movement.to_csv(out/"market_line_movement_v09.csv", index=False)

    result = {
        "version": "0.9",
        "rows": int(len(market)),
        "games": int(market["game_id"].nunique()),
        "sportsbooks": sorted(market["sportsbook"].dropna().astype(str).unique().tolist()),
        "phases": sorted(market["market_phase"].dropna().astype(str).unique().tolist()),
    }

    (out/"market_summary_v09.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
