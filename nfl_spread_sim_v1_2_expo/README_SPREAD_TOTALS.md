# Spread + game-total model add-on (research baseline)

This adds Python model code only. It does not replace `mobile/`, change Expo
packages, connect live odds, or modify the existing app API. Extract into the
existing `nfl_spread_sim_v1_2_expo` project root, next to `src/` and `mobile/`.

## What is implemented

- Two score targets: home final points and away final points.
- Both teams' pregame feature LEVELS, not just their differences. Equal
  offensive ratings alone cannot distinguish a high-scoring matchup from a
  low-scoring matchup.
- A regularized, multi-target Ridge baseline with train-only imputation/scaling.
- Season-ordered backtesting, separate margin and total MAE/RMSE, and comparison
  with a simple mean-score baseline fitted on each training fold.
- Paired out-of-sample home/away score residuals. For each simulation, draw the
  two errors from the SAME historical game, rather than independently.
- Exactly 1,000,000 paired score draws in the prediction command. Each draw
  yields both a margin and a combined total.
- Home cover/away cover/push and over/under/push probabilities.
- Price-aware model expected value, with pushes treated as refunds. Break-even
  comparisons use win probability conditional on the bet not pushing.
- Chronological interval diagnostics: a test season uses residuals only from
  earlier validation seasons, not from itself or later seasons.

## Important limitations

This is a tested SOFTWARE baseline, not a proven profitable betting model.
Real historical training was not run in the assistant's sandbox because the
historical data download was unavailable there. Tests use synthetic fixtures.

Only the existing recency-weighted TEAM features and opponent-adjustment
features enter this first dual-market baseline. Existing QB, college, stadium,
injury, weather, and opening-drive research modules remain untouched and are
NOT silently claimed as active inputs. We need to audit their source timing
and add them through controlled comparisons.

The original QB selection uses the player with most dropbacks in the game,
which is postgame information; it must not be used as a pregame starter source.
The existing opening-drive extraction has a first-20-plays fallback and score
measurement issues. That path is deliberately NOT included here.

The statistical score bootstrap clips negative scores and rounds to integers.
It reports the clipping fraction. This can shift simulated means relative to
raw regression means. It does NOT implement possession dynamics, NFL scoring
combinations, calibrated key-number masses, overtime rules, or a separate
learned game-specific variance model. Raw simulation probabilities are labeled
UNCALIBRATED. Use the reported simulation means for point differentials.

The old margin-only residual library cannot generate valid total probabilities.
This module requires its own paired-score model artifact; it rejects the old
artifact rather than inventing a total.

Training results from 2018-2025 do not make a current 2026 feature snapshot.
Predictions require a separately prepared, up-to-date pregame snapshot.
Historical PBP can also be revised: shifted features prevent current-game label
leakage, but do not prove that every upstream derived statistic existed in the
same form at the original prediction timestamp.

## Install without changing Expo

Open a SECOND Codespace terminal. Keep the Expo preview terminal running.

```bash
cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo
python -m venv .venv-model
source .venv-model/bin/activate
python -m pip install numpy pandas pyarrow scikit-learn joblib requests pytest
```

No `npm` commands are needed. Keep `.venv-model/` and model artifacts out of Git.

## Run the software tests

```bash
python -m pytest model_tests/test_dual_market.py -q
```

## Train with real historical data in Codespaces

```bash
python -m score_model.train --start-season 2018 --end-season 2025 --first-test-season 2022
```

This downloads/uses historical nflverse PBP through the existing loader, one
season at a time, builds the existing pregame team features, and writes:

- `artifacts/dual_market/model.joblib`
- `artifacts/dual_market/backtest_report.json`
- `artifacts/dual_market/walk_forward_predictions.csv`
- `artifacts/dual_market/training_games.csv`
- `artifacts/dual_market/team_games.csv`

The first validation season has no previous out-of-sample residual library, so
its interval diagnostic is intentionally unavailable, not fabricated.
The training command does NOT report ATS profitability, total-bet profitability,
or probability calibration. Those require verified timestamped historical lines
and another evaluation step. The data labels include final scores; match them
to each bookmaker's settlement rules before evaluating bets.

An existing prepared training-frame CSV containing all 28 home_pre_*/away_pre_*
columns can be passed with `--training-frame path/to/training.csv`. Target columns
are `home_score` and `away_score`; each row needs game_id, season and week.

## Future pregame inference (after training)

```bash
python -m score_model.predict --features data/verified_pregame_features.csv --home-spread -3.5 --total-line 46.5 --home-odds -110 --away-odds -110 --over-odds -110 --under-odds -110
```

The feature file above is a REQUIRED real input, not included fake data.
Do not substitute a final-result row or an actual current-game opening drive.
Only load joblib model files that you trust.

## Sources checked for this design

- nflverse/nflreadr schedule dictionary: final home/away scores, result and total.
- scikit-learn common pitfalls: preprocessing and data-leakage guidance.
- The Odds API v4: spreads and totals are separate supported markets; requesting
  both requires the market selector `spreads,totals`. No API key is bundled.

## Next controlled experiments

Add verified pregame QB and availability information; then historical offensive
pace/possessions, scoring efficiency, environment, and opening-drive tendencies.
Compare each addition on identical held-out games for BOTH targets. Test point
error, interval coverage, probability calibration and actual market-price
performance separately. One million draws reduce numerical sampling noise;
they do not create one million independent historical games.
