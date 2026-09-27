# Install v1.3 in your existing Codespace

From the project root:

```bash
unzip -n nfl_spread_totals_v1_3_market_backtester.zip
source .venv-model/bin/activate
python -m pytest market_tests/test_market_backtest_v13.py -q
```

Then prepare:

- `data/historical_market.csv`
- your real out-of-sample prediction file from the paired-score model

Run the backtester:

```bash
python -m score_model.historical_market \
  --market-csv data/historical_market.csv \
  --oos-predictions artifacts/dual_market/oos_predictions.csv \
  --output-dir artifacts/market_backtest_v13 \
  --min-edge 0.00
```
