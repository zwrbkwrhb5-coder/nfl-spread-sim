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
from .injuries import build_team_game_injury_features, merge_injury_features_into_games
from .model import walk_forward_predictions, summarize_validation, new_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stadiums", default="data/stadium_environment.csv")
    ap.add_argument("--availability", default="data/player_availability_template.csv")
    ap.add_argument("--start-season", type=int, default=2018)
    ap.add_argument("--end-season", type=int, default=2025)
    ap.add_argument("--first-test-season", type=int, default=2022)
    args = ap.parse_args()

    pbp = load_seasons(args.start_season, args.end_season)

    tg = build_team_games(pbp)
    tg = build_features(tg, halflife_games=6.0, min_games=3)
    games, team_features = build_matchups(tg)

    qg = add_qb_pregame_form(
        build_qb_games(pbp), halflife_games=5.0, min_games=2
    )
    games, qb_features = merge_qb_features_into_matchups(games, qg)

    stadiums = pd.read_csv(args.stadiums)
    games, stadium_features = attach_stadium_environment(games, stadiums)
    games, env_interactions = add_noise_interactions(games)

    availability = pd.read_csv(args.availability)
    team_inj = build_team_game_injury_features(availability)
    games, injury_features = merge_injury_features_into_games(games, team_inj)

    base_features = (
        team_features + qb_features + stadium_features + env_interactions
    )
    full_features = base_features + injury_features

    common = games.dropna(subset=full_features).copy()

    base_preds = walk_forward_predictions(
        common, base_features, args.first_test_season
    )
    full_preds = walk_forward_predictions(
        common, full_features, args.first_test_season
    )

    base_metrics = summarize_validation(base_preds)
    full_metrics = summarize_validation(full_preds)

    result = {
        "version": "0.6",
        "common_games": len(full_preds),
        "base_team_qb_environment": base_metrics,
        "with_player_availability": full_metrics,
        "injury_features": injury_features,
        "delta_mae": full_metrics["mae"] - base_metrics["mae"],
        "delta_rmse": full_metrics["rmse"] - base_metrics["rmse"],
    }

    out = Path("artifacts")
    out.mkdir(exist_ok=True)

    (out/"injury_comparison_v06.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    base_preds.to_csv(out/"v06_base_predictions.csv", index=False)
    full_preds.to_csv(out/"v06_injury_predictions.csv", index=False)

    model = new_model()
    model.fit(common[full_features], common["home_margin"])

    import joblib
    joblib.dump(
        {
            "version": "0.6",
            "model": model,
            "features": full_features,
            "base_features": base_features,
            "injury_features": injury_features,
            "residuals": full_preds["residual"].to_numpy(),
            "metrics": full_metrics,
            "comparison": result,
        },
        out/"spread_model_v06.joblib",
    )

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
