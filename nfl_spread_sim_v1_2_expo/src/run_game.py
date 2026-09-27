from __future__ import annotations
import argparse, json
import pandas as pd

from .pipeline import SpreadPipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True,
                    help="CSV with exactly one matchup feature row")
    ap.add_argument("--market", required=True)
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    args = ap.parse_args()

    X = pd.read_csv(args.features)
    if len(X) != 1:
        raise ValueError("Feature CSV must contain exactly one matchup row.")

    market = pd.read_csv(args.market)

    pipe = SpreadPipeline()
    result = pipe.analyze_books(
        X=X,
        market=market,
        game_id=args.game_id,
        home_team=args.home,
        away_team=args.away,
        scan_sims=250_000,
        final_sims=1_000_000,
    )

    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
