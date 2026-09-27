"""Train two score means; keep paired out-of-sample errors for simulation.

This is a statistical baseline, NOT a drive-by-drive NFL simulator. Integer
rounding/clipping is approximate and does not calibrate key scoring numbers.
No price, final score, actual starter, or opening-drive outcome is a feature.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ['home_score', 'away_score']
BASES = [
    'off_epa', 'def_epa', 'off_success', 'def_success',
    'explosive_pass_rate', 'sack_allowed_rate', 'def_sack_rate',
    'turnover_rate', 'points_for', 'points_against',
    'adj_off_epa', 'adj_def_epa', 'adj_off_success', 'adj_def_success',
]
FEATURES = [f'{side}_pre_{base}' for side in ('home', 'away') for base in BASES]


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ['game_id', 'season', 'week', *TARGETS, *FEATURES]
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError(f'Missing training columns: {missing}')
    x = frame.copy()
    if x.empty or x['game_id'].isna().any() or x['game_id'].duplicated().any():
        raise ValueError('Need a nonempty frame with unique game_id values.')
    for c in ['season', 'week', *TARGETS, *FEATURES]:
        x[c] = pd.to_numeric(x[c], errors='raise')
    if x[['season', 'week', *TARGETS]].isna().any().any():
        raise ValueError('Training requires completed games and known season/week.')
    y = x[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(y).all() or np.any(y < 0) or not np.all(y == np.round(y)):
        raise ValueError('Final scores must be finite, nonnegative integers.')
    if np.isinf(x[FEATURES].to_numpy(dtype=float)).any():
        raise ValueError('Infinite feature values are not allowed.')
    return x.sort_values(['season', 'week', 'game_id']).reset_index(drop=True)


def new_model(alpha: float = 100.0):
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be positive and finite.')
    # All preprocessing is fitted inside each training fold, not on test data.
    # Ridge supports multiple targets: column 0 = home score; column 1 = away.
    return make_pipeline(
        SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    def error(a, p):
        d = p - a
        return {'mae': float(np.mean(np.abs(d))),
                'rmse': float(np.sqrt(np.mean(d * d))),
                'bias_predicted_minus_actual': float(np.mean(d))}
    return {
        'games': int(len(actual)),
        'spread_margin': error(actual[:, 0] - actual[:, 1], predicted[:, 0] - predicted[:, 1]),
        'game_total': error(actual.sum(axis=1), predicted.sum(axis=1)),
    }


def score_pool(predicted_scores, residual_pairs) -> tuple[np.ndarray, float]:
    mean = np.asarray(predicted_scores, dtype=float)
    r = np.asarray(residual_pairs, dtype=float)
    if mean.shape != (2,) or not np.isfinite(mean).all() or np.any(mean < 0):
        raise ValueError('predicted_scores must be two nonnegative finite scores.')
    if r.ndim != 2 or r.shape[1] != 2 or len(r) < 100 or not np.isfinite(r).all():
        raise ValueError('Need >=100 finite PAIRED out-of-sample score errors, shape (n, 2).')
    # Preserve the home/away pairing from each historical game. The empirical
    # joint shape is retained before clipping/rounding; this is not calibration.
    raw = mean + r - r.mean(axis=0)
    clipped_fraction = float(np.mean(np.any(raw < 0, axis=1)))
    scores = np.rint(np.maximum(raw, 0)).astype(np.int32)
    return scores, clipped_fraction


def simulate_scores(predicted_scores, residual_pairs, n=1_000_000, seed=7):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 1 <= n <= 10_000_000:
        raise ValueError('n must be an integer between 1 and 10,000,000.')
    pool, clipped_fraction = score_pool(predicted_scores, residual_pairs)
    indexes = np.random.default_rng(seed).integers(0, len(pool), size=n)
    return pool[indexes], {'negative_score_clip_fraction_in_pool': clipped_fraction,
                          'residual_games': int(len(pool)), 'seed': int(seed)}


def decimal_odds(american: float) -> float:
    a = float(american)
    if not math.isfinite(a) or abs(a) < 100:
        raise ValueError('American odds must be finite and <= -100 or >= +100.')
    return 1 + (100 / -a if a < 0 else a / 100)


def grade(values: np.ndarray, odds: float) -> dict[str, Any]:
    win = float(np.mean(values > 0))
    push = float(np.mean(values == 0))
    loss = float(np.mean(values < 0))
    decimal = decimal_odds(odds)
    resolved = win + loss
    conditional = win / resolved if resolved > 0 else None
    break_even = 1 / decimal
    return {
        'win_probability': win, 'push_probability': push, 'loss_probability': loss,
        'win_probability_excluding_pushes': conditional,
        'break_even_excluding_pushes': break_even,
        'edge_excluding_pushes': None if conditional is None else conditional - break_even,
        'model_expected_profit_per_unit_staked': win * (decimal - 1) - loss,
    }


def analyze_scores(scores, home_spread: float, total_line: float,
                   home_odds=-110, away_odds=-110, over_odds=-110, under_odds=-110):
    a = np.asarray(scores)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) == 0:
        raise ValueError('scores must be a nonempty (n, 2) array.')
    if not np.isfinite(a).all() or np.any(a < 0) or not np.all(a == np.round(a)):
        raise ValueError('Simulation scores must be finite nonnegative integers.')
    if not math.isfinite(home_spread) or not math.isfinite(total_line) or total_line < 0:
        raise ValueError('Lines must be finite; the total cannot be negative.')
    # Integer/half-point lines only; no Asian-quarter settlement assumptions.
    if not float(home_spread * 2).is_integer() or not float(total_line * 2).is_integer():
        raise ValueError('Use whole-point or half-point lines.')
    margin, total = a[:, 0] - a[:, 1], a.sum(axis=1)
    mean_h, mean_a = a.mean(axis=0)
    return {
        'simulations_run': len(a), 'calibration_status': 'uncalibrated research baseline',
        'score_projection_from_simulation': {'home': float(mean_h), 'away': float(mean_a)},
        'spread': {
            'projected_home_margin': float(margin.mean()),
            'home_line': float(home_spread),
            'home_projected_differential': float(margin.mean() + home_spread),
            'home': grade(margin + home_spread, home_odds),
            'away': grade(-margin - home_spread, away_odds),
        },
        'total': {
            'projected_total': float(total.mean()), 'market_line': float(total_line),
            'over_projected_differential': float(total.mean() - total_line),
            'under_projected_differential': float(total_line - total.mean()),
            'over': grade(total - total_line, over_odds),
            'under': grade(total_line - total, under_odds),
        },
        'central_80_percent_intervals': {
            'home_margin': np.quantile(margin, [0.1, 0.9]).tolist(),
            'game_total': np.quantile(total, [0.1, 0.9]).tolist(),
        },
    }


def walk_forward(frame: pd.DataFrame, first_test_season: int, alpha=100.0):
    x = validate_frame(frame)
    predictions, prior_residuals = [], []
    for season in sorted(x.loc[x['season'] >= first_test_season, 'season'].unique()):
        train, test = x[x['season'] < season], x[x['season'] == season]
        if len(train) < 100:
            raise ValueError(f'Need at least 100 earlier games before testing {season}.')
        m = new_model(alpha)
        m.fit(train[FEATURES], train[TARGETS])
        p = np.maximum(0, m.predict(test[FEATURES]))
        actual = test[TARGETS].to_numpy(dtype=float)
        out = test[['game_id', 'season', 'week', *TARGETS]].copy()
        out['pred_home_score'], out['pred_away_score'] = p[:, 0], p[:, 1]
        out['residual_home'], out['residual_away'] = (actual - p).T
        out['trained_through_season'] = int(train['season'].max())
        baseline = train[TARGETS].mean().to_numpy(dtype=float)
        out['baseline_home_score'], out['baseline_away_score'] = baseline
        out['prior_residual_games'] = sum(len(r) for r in prior_residuals)
        # Distribution diagnostics are PREQUENTIAL: only errors from EARLIER
        # validation seasons may form the current season's residual library.
        for field in ['margin_80_covered', 'total_80_covered', 'margin_80_width', 'total_80_width']:
            out[field] = np.nan
        if out['prior_residual_games'].iloc[0] >= 100:
            residuals = np.vstack(prior_residuals)
            coverage = []
            for mean, truth in zip(p, actual):
                pool, _ = score_pool(mean, residuals)
                margin, total = pool[:, 0] - pool[:, 1], pool.sum(axis=1)
                qm, qt = np.quantile(margin, [.1, .9]), np.quantile(total, [.1, .9])
                coverage.append([qm[0] <= truth[0]-truth[1] <= qm[1],
                                 qt[0] <= truth.sum() <= qt[1], qm[1]-qm[0], qt[1]-qt[0]])
            out[['margin_80_covered', 'total_80_covered', 'margin_80_width', 'total_80_width']] = np.asarray(coverage, dtype=float)
        predictions.append(out)
        prior_residuals.append(actual - p)
    if not predictions:
        raise ValueError('No test seasons available.')
    pred = pd.concat(predictions, ignore_index=True)
    actual = pred[TARGETS].to_numpy(dtype=float)
    means = pred[['pred_home_score', 'pred_away_score']].to_numpy(dtype=float)
    baseline = pred[['baseline_home_score', 'baseline_away_score']].to_numpy(dtype=float)
    diagnostics = pred.dropna(subset=['margin_80_covered', 'total_80_covered'])
    report = {
        'model': 'paired-score Ridge baseline; alpha not yet tuned',
        'point_forecast': point_metrics(actual, means),
        'prior_training_mean_baseline': point_metrics(actual, baseline),
        'distribution_test_games': int(len(diagnostics)),
        'interval_diagnostics': {c: (float(diagnostics[c].mean()) if len(diagnostics) else None)
            for c in ['margin_80_covered', 'total_80_covered', 'margin_80_width', 'total_80_width']},
        'ats_and_over_under_profit_tested': False,
        'probability_calibration_fitted': False,
    }
    report['by_test_season'] = {
        str(int(s)): point_metrics(g[TARGETS].to_numpy(dtype=float),
            g[['pred_home_score', 'pred_away_score']].to_numpy(dtype=float))
        for s, g in pred.groupby('season')
    }
    return pred, report
