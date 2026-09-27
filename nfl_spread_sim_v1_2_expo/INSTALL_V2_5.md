# v2.5 Totals Context Market Validation

This is the market-validation gate for the v2.4 context/weather result.

Configuration:

- Spread: unchanged QB-enhanced model
- Totals baseline: QB-enhanced totals model
- Totals candidate: QB-enhanced totals model + context/weather

The baseline and candidate use:
- the exact same historical training rows
- the exact same OOS test games
- identical spread predictions

Only the totals model changes.

## 1. Install + test

```bash
unzip -n nfl_spread_totals_v2_5_totals_context_market.zip
source .venv-model/bin/activate
python -m pytest context_tests/test_totals_context_v25.py -q
```

## 2. Generate common-sample walk-forward OOS predictions

```bash
python -m score_model.totals_context_oos_v25 \
  --training-csv artifacts/dual_market/training_games_qb_context_v24.csv \
  --first-test-season 2022 \
  --baseline-output artifacts/dual_market/oos_qb_total_base_v25.csv \
  --context-output artifacts/dual_market/oos_qb_total_context_v25.csv
```

Expected confirmation:
- `Common sample: true`
- `Spread predictions identical: true`

## 3. Run both through the week-safe market backtest

```bash
python -m score_model.compare_totals_context_market_v25 \
  --market-csv data/historical_market.csv \
  --baseline-predictions artifacts/dual_market/oos_qb_total_base_v25.csv \
  --context-predictions artifacts/dual_market/oos_qb_total_context_v25.csv \
  --output artifacts/dual_market/totals_context_market_v25.json
```

## 4. View the final comparison

```bash
python -m json.tool artifacts/dual_market/totals_context_market_v25.json
```

Key fields:

- `total_roi_context_minus_base`
  - positive = context improved historical total-bet ROI

- `total_brier_context_minus_base`
  - negative = context improved probability calibration

- `total_win_rate_context_minus_base`
  - positive = context improved settled-bet win rate

The spread deltas should be zero or effectively zero because the spread
predictions are intentionally unchanged.

Promote context/weather into the live totals model only if the week-safe
market comparison supports it.
