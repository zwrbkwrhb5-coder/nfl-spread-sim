# v1.5 Leakage-Safe Validation

## Install / test

```bash
unzip -n nfl_spread_totals_v1_5_leakage_safe.zip
source .venv-model/bin/activate
python -m pytest market_tests/test_v15_leakage_safe.py -q
```

## Run clean walk-forward backtest

```bash
python -m score_model.market_backtest_v15 \
  --market-csv data/historical_market.csv \
  --oos-predictions artifacts/dual_market/oos_predictions.csv \
  --output-dir artifacts/market_backtest_v15 \
  --min-edge 0.00
```

## View main report

```bash
python -m json.tool artifacts/market_backtest_v15/report.json
```

## Extra diagnostics

Outputs include:

- summary_by_market.csv
- summary_by_season.csv
- summary_by_edge_bucket.csv
- spread_home_away.csv
- spread_by_line_size.csv
- best_per_game_market.csv
- bettable.csv

## Why v1.5 matters

v1.4 used the full 2022–2025 out-of-sample residual pool for historical
probability estimates. v1.5 removes that leakage.

For every game, residual distributions and calibration are built only from
games that occurred earlier in the walk-forward sequence.
