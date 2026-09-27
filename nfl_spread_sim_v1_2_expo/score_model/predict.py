"""Research CLI. Supply an explicitly pregame feature snapshot, never a label row."""
from __future__ import annotations
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from .core import FEATURES, analyze_scores, simulate_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='artifacts/dual_market/model.joblib')
    ap.add_argument('--features', required=True, help='One-row CSV of pregame home_pre_*/away_pre_* features.')
    ap.add_argument('--home-spread', type=float, required=True)
    ap.add_argument('--total-line', type=float, required=True)
    for name in ['home', 'away', 'over', 'under']:
        ap.add_argument(f'--{name}-odds', type=int, default=-110)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    # Only load a model file you trust; joblib/pickle can execute code.
    bundle = joblib.load(args.model)
    if bundle.get('targets') != ['home_score', 'away_score'] or bundle.get('features') != FEATURES:
        ap.error('This is not a compatible dual-market model. A margin-only model cannot be used.')
    x = pd.read_csv(args.features)
    if len(x) != 1 or any(c not in x for c in FEATURES):
        ap.error('Need exactly one pregame feature row containing all expected features.')
    x = x[FEATURES].apply(pd.to_numeric, errors='raise')
    if np.isinf(x.to_numpy(dtype=float)).any():
        ap.error('Infinite features are not valid.')
    mean = np.maximum(0, bundle['model'].predict(x)[0])
    scores, diagnostics = simulate_scores(mean, bundle['residual_pairs'], n=1_000_000, seed=args.seed)
    result = analyze_scores(scores, args.home_spread, args.total_line,
        args.home_odds, args.away_odds, args.over_odds, args.under_odds)
    result['statistical_mean_scores_before_integer_simulation'] = mean.tolist()
    result['training_through_season'] = bundle['trained_through_season']
    result['simulation_diagnostics'] = diagnostics
    result['warning'] = ('Uncalibrated research output. Verify that your pregame feature snapshot '
        'is current and was built without future outcomes. This does not fetch odds or injuries.')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
