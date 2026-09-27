from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib

from .data import load_seasons
from .features import build_team_games, build_features, build_matchups
from .model import new_model, walk_forward_predictions, summarize_validation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-season", type=int, default=2018)
    ap.add_argument("--end-season", type=int, default=2025)
    ap.add_argument("--first-test-season", type=int, default=2022)
    ap.add_argument("--halflife-games", type=float, default=6.0)
    ap.add_argument("--min-games", type=int, default=3)
    args = ap.parse_args()

    print("Loading play-by-play...")
    pbp = load_seasons(args.start_season, args.end_season)

    print("Building team-game aggregates...")
    tg = build_team_games(pbp)

    print("Adding recency and opponent-adjusted features...")
    tg = build_features(
        tg,
        halflife_games=args.halflife_games,
        min_games=args.min_games,
    )
    games, features = build_matchups(tg)

    print("Running walk-forward validation...")
    preds = walk_forward_predictions(games, features, args.first_test_season)
    metrics = summarize_validation(preds)

    print(json.dumps(metrics, indent=2))

    model = new_model()
    model.fit(games[features], games["home_margin"])

    out = Path("artifacts")
    out.mkdir(exist_ok=True)

    joblib.dump(
        {
            "version": "0.2",
            "model": model,
            "features": features,
            "residuals": preds["residual"].to_numpy(),
            "metrics": metrics,
            "train_start": args.start_season,
            "train_end": args.end_season,
            "halflife_games": args.halflife_games,
            "min_games": args.min_games,
        },
        out / "spread_model.joblib"
    )

    preds.to_csv(out / "walk_forward_predictions.csv", index=False)
    games.to_parquet(out / "training_games.parquet", index=False)
    tg.to_parquet(out / "team_games.parquet", index=False)

    print("Saved v0.2 model artifacts.")

if __name__ == "__main__":
    main()
