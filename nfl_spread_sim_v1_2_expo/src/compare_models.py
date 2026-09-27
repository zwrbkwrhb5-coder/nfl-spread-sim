from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

from .data import load_seasons
from .features import (
    build_team_games,
    build_features,
    build_matchups,
    merge_qb_features_into_matchups,
)
from .qb import build_qb_games, add_qb_pregame_form
from .model import new_model, walk_forward_predictions, summarize_validation

def run_model(df, features, first_test_season):
    preds = walk_forward_predictions(df.dropna(subset=features), features, first_test_season)
    metrics = summarize_validation(preds)
    return preds, metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-season", type=int, default=2018)
    ap.add_argument("--end-season", type=int, default=2025)
    ap.add_argument("--first-test-season", type=int, default=2022)
    ap.add_argument("--team-halflife", type=float, default=6.0)
    ap.add_argument("--qb-halflife", type=float, default=5.0)
    ap.add_argument("--min-team-games", type=int, default=3)
    ap.add_argument("--min-qb-games", type=int, default=2)
    args = ap.parse_args()

    print("Loading PBP...")
    pbp = load_seasons(args.start_season, args.end_season)

    print("Building team model rows...")
    tg = build_team_games(pbp)
    tg = build_features(
        tg,
        halflife_games=args.team_halflife,
        min_games=args.min_team_games,
    )
    games, team_features = build_matchups(tg)

    print("Building QB form rows...")
    qg = build_qb_games(pbp)
    qg = add_qb_pregame_form(
        qg,
        halflife_games=args.qb_halflife,
        min_games=args.min_qb_games,
    )

    print("Merging QB matchup features...")
    games_qb, qb_features = merge_qb_features_into_matchups(games, qg)

    print("Running team-only walk-forward...")
    team_preds, team_metrics = run_model(
        games,
        team_features,
        args.first_test_season,
    )

    print("Running team+QB walk-forward...")
    full_features = team_features + qb_features
    qb_ready = games_qb.dropna(subset=full_features).copy()
    qb_preds, qb_metrics = run_model(
        qb_ready,
        full_features,
        args.first_test_season,
    )

    # Apples-to-apples subset: games where both models can be evaluated.
    common_ids = set(qb_preds["game_id"])
    common_team = team_preds[team_preds["game_id"].isin(common_ids)].copy()

    # Recompute team-only metrics on exact same games.
    common_team_metrics = summarize_validation(common_team)

    result = {
        "team_only_all_games": team_metrics,
        "team_only_common_games": common_team_metrics,
        "team_plus_qb_common_games": qb_metrics,
        "qb_features": qb_features,
        "common_games": len(common_ids),
    }

    result["delta_mae_vs_common_team"] = (
        qb_metrics["mae"] - common_team_metrics["mae"]
    )
    result["delta_rmse_vs_common_team"] = (
        qb_metrics["rmse"] - common_team_metrics["rmse"]
    )

    print(json.dumps(result, indent=2))

    out = Path("artifacts")
    out.mkdir(exist_ok=True)

    team_preds.to_csv(out / "team_only_walk_forward.csv", index=False)
    qb_preds.to_csv(out / "team_qb_walk_forward.csv", index=False)
    games_qb.to_parquet(out / "training_games_with_qb.parquet", index=False)
    qg.to_parquet(out / "qb_games.parquet", index=False)

    Path(out / "model_comparison.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    # Train and save final team+QB model for later simulation use.
    final_df = qb_ready.copy()
    model = new_model()
    model.fit(final_df[full_features], final_df["home_margin"])

    import joblib
    joblib.dump(
        {
            "version": "0.4",
            "model": model,
            "team_features": team_features,
            "qb_features": qb_features,
            "features": full_features,
            "team_halflife": args.team_halflife,
            "qb_halflife": args.qb_halflife,
            "min_team_games": args.min_team_games,
            "min_qb_games": args.min_qb_games,
            "residuals": qb_preds["residual"].to_numpy(),
            "metrics": qb_metrics,
            "comparison": result,
        },
        out / "spread_model_v04.joblib",
    )

    print("Saved artifacts/spread_model_v04.joblib")
    print("Saved artifacts/model_comparison.json")

if __name__ == "__main__":
    main()
