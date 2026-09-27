"""Run from the existing Python project root, NOT its mobile directory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from .core import FEATURES, TARGETS, new_model, validate_frame, walk_forward


def main():
    ap = argparse.ArgumentParser(description='Train and backtest spread + total score baseline.')
    ap.add_argument('--start-season', type=int, default=2018)
    ap.add_argument('--end-season', type=int, default=2025)
    ap.add_argument('--first-test-season', type=int, default=2022)
    ap.add_argument('--training-frame', help='Optional CSV produced by existing build_matchups.')
    ap.add_argument('--output-dir', default='artifacts/dual_market')
    args = ap.parse_args()
    if not args.start_season < args.first_test_season <= args.end_season:
        ap.error('Require start-season < first-test-season <= end-season.')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.training_frame:
        games = pd.read_csv(args.training_frame)
        games = games[games['season'].between(args.start_season, args.end_season)]
    else:
        from src.data import load_pbp
        from src.features import build_team_games, build_features, build_matchups
        teams = []
        for year in range(args.start_season, args.end_season + 1):
            print(f'Loading and aggregating {year}...', flush=True)
            # One season at a time to avoid retaining every play of every season.
            pbp = load_pbp(year)
            teams.append(build_team_games(pbp))
            del pbp
        tg = build_features(pd.concat(teams, ignore_index=True), halflife_games=6, min_games=3)
        games, _ = build_matchups(tg)
        tg.to_csv(out / 'team_games.csv', index=False)
    games = validate_frame(games)
    preds, report = walk_forward(games, args.first_test_season)
    model = new_model()
    model.fit(games[FEATURES], games[TARGETS])
    bundle = {
        'version': 'dual-market-research-0.1', 'model': model, 'features': FEATURES,
        'targets': TARGETS,
        'residual_pairs': preds[['residual_home', 'residual_away']].to_numpy(),
        'trained_through_season': int(games['season'].max()),
        'team_halflife_games': 6, 'report': report,
        'active_layers': ['recency-weighted team metrics', 'existing opponent adjustments'],
        'inactive_layers': ['QB', 'college priors', 'injury return', 'stadium loudness',
                            'roster availability', 'weather', 'opening drives', 'live odds'],
    }
    joblib.dump(bundle, out / 'model.joblib')
    games.to_csv(out / 'training_games.csv', index=False)
    preds.to_csv(out / 'walk_forward_predictions.csv', index=False)
    (out / 'backtest_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f'Saved {out}. Research baseline only; no live app files were changed.')


if __name__ == '__main__':
    main()
