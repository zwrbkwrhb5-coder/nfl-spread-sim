from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

from .data import load_seasons
from .features import build_team_games, build_features, build_matchups, merge_qb_features_into_matchups
from .qb import build_qb_games, add_qb_pregame_form
from .opening_drive import (
    identify_opening_drives,
    add_opening_drive_pregame_form,
    build_opening_drive_defense_form,
    merge_opening_drive_features,
    merge_opening_drive_defense,
)
from .opening_drive_ats import (
    build_home_opening_drive_ats_study,
    summarize_opening_drive_effect,
)
from .model import walk_forward_predictions, summarize_validation, new_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-season", type=int, default=2018)
    ap.add_argument("--end-season", type=int, default=2025)
    ap.add_argument("--first-test-season", type=int, default=2022)
    args = ap.parse_args()

    pbp = load_seasons(args.start_season, args.end_season)

    tg = build_team_games(pbp)
    tg = build_features(tg, halflife_games=6.0, min_games=3)
    games, team_features = build_matchups(tg)

    qg = add_qb_pregame_form(build_qb_games(pbp), halflife_games=5.0, min_games=2)
    games, qb_features = merge_qb_features_into_matchups(games, qg)

    od = identify_opening_drives(pbp)
    od_form = add_opening_drive_pregame_form(od, halflife_games=6.0, min_games=3)

    game_teams = games[["game_id","home_team","away_team"]].drop_duplicates()
    od_def = build_opening_drive_defense_form(
        od, game_teams, halflife_games=6.0, min_games=3
    )

    games_od, od_off_features = merge_opening_drive_features(games, od_form)
    games_od, od_def_features = merge_opening_drive_defense(games_od, od_def)

    base_features = team_features + qb_features
    full_features = base_features + od_off_features + od_def_features

    common = games_od.dropna(subset=full_features).copy()

    base_preds = walk_forward_predictions(common, base_features, args.first_test_season)
    full_preds = walk_forward_predictions(common, full_features, args.first_test_season)

    base_metrics = summarize_validation(base_preds)
    full_metrics = summarize_validation(full_preds)

    study = build_home_opening_drive_ats_study(games, od)
    study_summary = summarize_opening_drive_effect(study)

    result = {
        "version": "0.10",
        "common_games": len(full_preds),
        "base_team_plus_qb": base_metrics,
        "with_opening_drive_features": full_metrics,
        "opening_drive_offense_features": od_off_features,
        "opening_drive_defense_features": od_def_features,
        "delta_mae": full_metrics["mae"] - base_metrics["mae"],
        "delta_rmse": full_metrics["rmse"] - base_metrics["rmse"],
    }

    out = Path("artifacts")
    out.mkdir(exist_ok=True)

    (out/"opening_drive_comparison_v10.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    study.to_csv(out/"opening_drive_home_study_v10.csv", index=False)
    study_summary.to_csv(out/"opening_drive_home_summary_v10.csv", index=False)
    base_preds.to_csv(out/"v10_base_predictions.csv", index=False)
    full_preds.to_csv(out/"v10_opening_drive_predictions.csv", index=False)

    model = new_model()
    model.fit(common[full_features], common["home_margin"])

    import joblib
    joblib.dump({
        "version":"0.10",
        "model":model,
        "features":full_features,
        "opening_drive_features":od_off_features + od_def_features,
        "residuals":full_preds["residual"].to_numpy(),
        "metrics":full_metrics,
        "comparison":result,
    }, out/"spread_model_v10.joblib")

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
