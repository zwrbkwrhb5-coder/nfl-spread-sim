import json
import numpy as np
import pandas as pd
import pytest
from score_model.core import (
    BASES, FEATURES, analyze_scores, decimal_odds, simulate_scores,
    score_pool, validate_frame, walk_forward,
)


def example_frame():
    rng = np.random.default_rng(13)
    n = 640
    x = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    x['game_id'] = [f'fixture-{i}' for i in range(n)]
    x['season'] = np.repeat([2020, 2021, 2022, 2023], 160)
    x['week'] = np.tile(np.arange(160) // 10 + 1, 4)
    x['home_score'] = np.maximum(0, np.round(24 + 3*x[FEATURES[0]] + rng.normal(0, 8, n)))
    x['away_score'] = np.maximum(0, np.round(21 + 2*x[FEATURES[14]] + rng.normal(0, 8, n)))
    return x


def test_two_offensive_levels_not_just_differences():
    # Both high-vs-high and low-vs-low have zero difference. Keep their levels.
    assert 'home_pre_points_for' in FEATURES
    assert 'away_pre_points_for' in FEATURES
    assert len(FEATURES) == 28
    assert not set(FEATURES) & {'home_score','away_score','home_margin','total'}


def test_shared_simulation_score_arithmetic():
    scores = np.array([[27, 20], [21, 24], [24, 21], [20, 20]])
    result = analyze_scores(scores, -3, 47)
    assert result['spread']['home']['win_probability'] == .25
    assert result['spread']['home']['push_probability'] == .25
    assert result['spread']['away']['win_probability'] == .5
    assert result['total']['over']['win_probability'] == 0
    assert result['total']['over']['push_probability'] == .25
    assert result['total']['under']['win_probability'] == .75
    assert result['total']['projected_total'] == 44.25
    assert result['spread']['projected_home_margin'] == 1.75
    json.dumps(result, allow_nan=False)


def test_half_point_lines_no_push():
    result = analyze_scores(np.array([[27, 20], [21, 24]]), -3.5, 46.5)
    assert result['spread']['home']['push_probability'] == 0
    assert result['total']['over']['push_probability'] == 0


def test_all_pushes_no_conditional_win_or_edge():
    result = analyze_scores(np.tile([24, 21], (100, 1)), -3, 45)
    for market, side in [('spread','home'), ('total','over')]:
        r = result[market][side]
        assert r['push_probability'] == 1
        assert r['win_probability_excluding_pushes'] is None
        assert r['edge_excluding_pushes'] is None
        assert r['model_expected_profit_per_unit_staked'] == 0


def test_pairing_is_retained():
    r = np.column_stack([np.linspace(-6, 6, 1000)] * 2)
    scores, _ = simulate_scores([24, 21], r, n=10000)
    # Independent resampling of teams would break this exact constant margin.
    assert np.all(scores[:, 0] - scores[:, 1] == 3)


def test_million_and_repeatable():
    rng = np.random.default_rng(5)
    errors = rng.normal(size=(500, 2))*5
    a, _ = simulate_scores([24, 21], errors)
    b, _ = simulate_scores([24, 21], errors)
    assert a.shape == (1_000_000, 2)
    assert np.array_equal(a, b)
    r = analyze_scores(a, -3, 45)
    for market, sides in [('spread', ['home','away']), ('total', ['over','under'])]:
        first, second = [r[market][s] for s in sides]
        assert first['win_probability'] == second['loss_probability']
        assert first['push_probability'] == second['push_probability']
        assert abs(sum(first[k] for k in ['win_probability','loss_probability','push_probability'])-1) < 1e-12


def test_clipping_is_disclosed_and_never_negative():
    r = np.tile([[-20, 0], [20, 0]], (100, 1))
    pool, clipped = score_pool([3, 3], r)
    assert np.min(pool) >= 0
    assert clipped == .5


@pytest.mark.parametrize('odds', [0, -99, 99, np.nan, np.inf])
def test_invalid_odds_fail(odds):
    with pytest.raises(ValueError):
        decimal_odds(odds)


def test_prices_and_pushes_in_expected_value():
    # 50% win, 10% push, 40% loss at -110.
    scores = np.array([[24, 20]]*50 + [[23, 20]]*10 + [[20, 23]]*40)
    r = analyze_scores(scores, -3, 45)['spread']['home']
    assert r['model_expected_profit_per_unit_staked'] == pytest.approx(.5*100/110-.4)
    assert r['win_probability_excluding_pushes'] == pytest.approx(.5/.9)


@pytest.mark.parametrize('line', [np.nan, np.inf, 3.25])
def test_bad_lines_fail(line):
    with pytest.raises(ValueError):
        analyze_scores(np.array([[24, 21]]), line, 45)


@pytest.mark.parametrize('n', [0, -1, True, 1.5, 10_000_001])
def test_bad_simulation_count(n):
    with pytest.raises(ValueError):
        simulate_scores([24, 21], np.zeros((100, 2)), n=n)


def test_no_future_season_predictions_or_residuals():
    x = example_frame()
    p, report = walk_forward(x, first_test_season=2022)
    assert np.all(p['trained_through_season'] < p['season'])
    assert (p.loc[p['season'] == 2022, 'prior_residual_games'] == 0).all()
    assert (p.loc[p['season'] == 2023, 'prior_residual_games'] == 160).all()
    assert report['distribution_test_games'] == 160
    assert report['point_forecast']['games'] == 320
    x.loc[x['season'] == 2023, ['home_score', 'away_score']] = [99, 0]
    changed, _ = walk_forward(x, first_test_season=2022)
    # Current test outcomes cannot change predicted score means for that season.
    assert np.allclose(p[['pred_home_score','pred_away_score']], changed[['pred_home_score','pred_away_score']])
    json.dumps(report, allow_nan=False)


def test_duplicate_game_rejected():
    x = example_frame()
    with pytest.raises(ValueError, match='unique'):
        validate_frame(pd.concat([x, x.iloc[:1]]))


def test_upstream_pregame_features_exclude_current_outcomes():
    from src.features import build_features, build_matchups
    rows = []
    for week in range(1, 12):
        for side, team, opp in [(1,'A','B'), (0,'B','A')]:
            row = dict(game_id=f'g{week:02d}', season=2024, week=week,
                team=team, opponent=opp, is_home=side, home_score=21, away_score=20)
            row.update({c: float(week) for c in BASES if not c.startswith('adj_')})
            rows.append(row)
    raw = pd.DataFrame(rows)
    a, _ = build_matchups(build_features(raw, min_games=1))
    raw.loc[raw.week == 11, ['off_epa','points_for','points_against']] = 10000
    b, _ = build_matchups(build_features(raw, min_games=1))
    pa, pb = a[a.week == 11], b[b.week == 11]
    assert np.allclose(pa[FEATURES], pb[FEATURES])
