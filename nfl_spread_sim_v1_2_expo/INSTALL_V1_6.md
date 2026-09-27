# v1.6 Week-Safe Backtest + Feature Ablation

## Install / test

```bash
unzip -n nfl_spread_totals_v1_6_week_safe_ablation.zip
source .venv-model/bin/activate
python -m pytest market_tests/test_v16_week_safe.py -q
```

## Run the stricter backtest

```bash
python -m score_model.market_backtest_v16 \
  --market-csv data/historical_market.csv \
  --oos-predictions artifacts/dual_market/walk_forward_predictions.csv \
  --output-dir artifacts/market_backtest_v16 \
  --min-edge 0.00
```

View:

```bash
python -m json.tool artifacts/market_backtest_v16/report.json
```

## Feature ablation

The included ablation harness tests each feature group added to the base model
and reports the change in walk-forward MAE / RMSE for margin and total.

It only works for feature columns that are actually present in
`artifacts/dual_market/training_games.csv`.

Run:

```bash
python -m score_model.feature_ablation_v16 \
  --training-csv artifacts/dual_market/training_games.csv \
  --groups-json config/feature_groups_v16.json \
  --first-test-season 2022
```

If a proposed feature column does not exist yet, add/populate that feature in
the training table before running the group.
