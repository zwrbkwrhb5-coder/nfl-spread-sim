# v1.8 QB-Enhanced Market Validation

## 1. Install + test

```bash
unzip -n nfl_spread_totals_v1_8_qb_market_validation.zip
source .venv-model/bin/activate
python -m pytest qb_tests/test_qb_v18.py -q
```

## 2. Tune QB shrinkage

```bash
python -m score_model.qb_tuned_v18 \
  --training-csv artifacts/dual_market/training_games_qb_v17.csv \
  --output artifacts/dual_market/qb_tuning_v18.json
```

View:

```bash
python -m json.tool artifacts/dual_market/qb_tuning_v18.json
```

Use the chosen shrinkage value for the next step.

## 3. Generate fresh walk-forward OOS predictions

Example with 300 dropbacks:

```bash
python -m score_model.qb_oos_v18 \
  --training-csv artifacts/dual_market/training_games_qb_v17.csv \
  --shrinkage-dropbacks 300 \
  --baseline-output artifacts/dual_market/oos_baseline_v18.csv \
  --qb-output artifacts/dual_market/oos_qb_v18.csv
```

## 4. Compare baseline vs QB on week-safe market results

```bash
python -m score_model.compare_market_v18 \
  --market-csv data/historical_market.csv \
  --baseline-predictions artifacts/dual_market/oos_baseline_v18.csv \
  --qb-predictions artifacts/dual_market/oos_qb_v18.csv \
  --output artifacts/dual_market/qb_market_comparison_v18.json
```

## 5. View final comparison

```bash
python -m json.tool artifacts/dual_market/qb_market_comparison_v18.json
```

Interpretation:
- positive spread/total ROI delta = QB improved market ROI
- negative Brier delta = QB improved probability quality

Keep QB only if it improves enough metrics to justify the added complexity.
