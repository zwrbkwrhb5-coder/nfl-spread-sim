# v1.7 QB Layer

This release adds real historical QB pregame features to the spread + totals
training table and A/B tests them against the current baselines.

## QB metrics

- EPA / dropback
- success rate
- CPOE
- sack rate
- interception rate
- explosive-pass rate
- prior starts
- prior dropbacks

Starter approximation:
the QB with the most dropbacks for that team in that historical game.

All QB form features are shifted before the EWM calculation, so the current
game cannot enter its own pregame QB features.

## 1. Install and test

```bash
unzip -n nfl_spread_totals_v1_7_qb_layer.zip
source .venv-model/bin/activate
python -m pytest qb_tests/test_qb_layer_v17.py -q
```

## 2. Build QB-enhanced historical training data

```bash
python -m score_model.qb_layer_v17 \
  --training-csv artifacts/dual_market/training_games_ablation.csv \
  --start-season 2018 \
  --end-season 2025 \
  --halflife-games 5 \
  --min-games 2 \
  --output artifacts/dual_market/training_games_qb_v17.csv
```

This downloads nflverse play-by-play parquet files for 2018–2025.

## 3. Run the QB A/B test

```bash
python -m score_model.qb_ablation_v17 \
  --training-csv artifacts/dual_market/training_games_qb_v17.csv \
  --first-test-season 2022 \
  --output artifacts/dual_market/qb_ablation_v17.json
```

## 4. View results

```bash
python -m json.tool artifacts/dual_market/qb_ablation_v17.json
```

Interpretation:

- negative `delta_mae_qb_minus_base` = QB features improved MAE
- negative `delta_rmse_qb_minus_base` = QB features improved RMSE

The comparison uses the exact same games for base vs base+QB, so missing QB
history cannot make one model look better simply by changing the sample.

## Important

This is a feature-ablation test, not proof of betting profitability.
If QB improves forecast accuracy, the next step is to regenerate OOS score
predictions with QB included and rerun the week-safe market backtest.
