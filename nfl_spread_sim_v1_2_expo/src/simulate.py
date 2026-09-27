from __future__ import annotations
import argparse, json
import joblib
import pandas as pd

from .predict import matchup_vector
from .sim_engine import simulate_spread

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--spread", required=True, type=float,
                    help="HOME spread. Example: -3.5 means home favored by 3.5")
    ap.add_argument("--odds", type=int, default=-110)
    ap.add_argument("--simulations", type=int, default=1_000_000)
    args = ap.parse_args()

    bundle = joblib.load("artifacts/spread_model.joblib")
    tg = pd.read_parquet("artifacts/team_games.parquet")

    X = matchup_vector(
        tg,
        args.home,
        args.away,
        bundle["features"],
        halflife_games=bundle.get("halflife_games", 6.0),
        min_games=bundle.get("min_games", 3),
    )
    pred = float(bundle["model"].predict(X)[0])

    result = simulate_spread(
        projected_home_margin=pred,
        residuals=bundle["residuals"],
        home_spread=args.spread,
        odds=args.odds,
        n=args.simulations,
    )

    result["model_version"] = bundle.get("version", "unknown")
    result["home_team"] = args.home
    result["away_team"] = args.away
    result["validation_metrics"] = bundle["metrics"]

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
