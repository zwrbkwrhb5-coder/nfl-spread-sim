from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

from .data import load_seasons
from .features import (
    build_team_games, build_features, build_matchups,
    merge_qb_features_into_matchups,
)
from .qb import build_qb_games, add_qb_pregame_form
from .stadium import attach_stadium_environment, add_noise_interactions
from .model import walk_forward_predictions, summarize_validation, new_model

def evaluate(df, features, first_test):
    ready = df.dropna(subset=features).copy()
    p = walk_forward_predictions(ready, features, first_test)
    return p, summarize_validation(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stadiums", default="data/stadium_environment.csv")
    ap.add_argument("--start-season", type=int, default=2018)
    ap.add_argument("--end-season", type=int, default=2025)
    ap.add_argument("--first-test-season", type=int, default=2022)
    ap.add_argument("--team-halflife", type=float, default=6.0)
    ap.add_argument("--qb-halflife", type=float, default=5.0)
    args = ap.parse_args()

    pbp = load_seasons(args.start_season, args.end_season)
    tg = build_team_games(pbp)
    tg = build_features(tg, halflife_games=args.team_halflife, min_games=3)
    games, team_features = build_matchups(tg)

    qg = add_qb_pregame_form(
        build_qb_games(pbp), halflife_games=args.qb_halflife, min_games=2
    )
    games, qb_features = merge_qb_features_into_matchups(games, qg)

    stadiums = pd.read_csv(args.stadiums)
    games, stadium_features = attach_stadium_environment(games, stadiums)
    games, interaction_features = add_noise_interactions(games)

    base_features = team_features + qb_features
    env_features = base_features + stadium_features + interaction_features

    # Force exact common sample for fair A/B comparison.
    common = games.dropna(subset=env_features).copy()
    base_preds = walk_forward_predictions(common, base_features, args.first_test_season)
    env_preds = walk_forward_predictions(common, env_features, args.first_test_season)

    base_metrics = summarize_validation(base_preds)
    env_metrics = summarize_validation(env_preds)

    result = {
        "version": "0.5",
        "common_games": len(env_preds),
        "base_team_plus_qb": base_metrics,
        "team_qb_plus_environment": env_metrics,
        "stadium_features": stadium_features,
        "interaction_features": interaction_features,
        "delta_mae": env_metrics["mae"] - base_metrics["mae"],
        "delta_rmse": env_metrics["rmse"] - base_metrics["rmse"],
    }

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    (out/"stadium_comparison_v05.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    base_preds.to_csv(out/"v05_base_predictions.csv", index=False)
    env_preds.to_csv(out/"v05_environment_predictions.csv", index=False)

    model = new_model()
    model.fit(common[env_features], common["home_margin"])

    import joblib
    joblib.dump({
        "version": "0.5",
        "model": model,
        "features": env_features,
        "team_features": team_features,
        "qb_features": qb_features,
        "stadium_features": stadium_features,
        "interaction_features": interaction_features,
        "residuals": env_preds["residual"].to_numpy(),
        "metrics": env_metrics,
        "comparison": result,
    }, out/"spread_model_v05.joblib")

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
