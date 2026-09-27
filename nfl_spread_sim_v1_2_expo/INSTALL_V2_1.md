# v2.1 Split Injury Market Validation

This version tests the market-specific injury configuration:

- Spread: QB baseline + player-value injuries
- Totals: QB baseline + basic injury burden

Both are compared to the QB-only baseline on the exact same game set.

## 1. Install + test

```bash
unzip -n nfl_spread_totals_v2_1_split_injury_market.zip
source .venv-model/bin/activate
python -m pytest injury_tests/test_split_v21.py -q
```

## 2. Generate common-sample OOS predictions

```bash
python -m score_model.split_injury_oos_v21 \
  --training-csv artifacts/dual_market/training_games_qb_injury_pv_v20.csv \
  --first-test-season 2022 \
  --max-test-season 2024 \
  --qb-output artifacts/dual_market/oos_qb_common_v21.csv \
  --split-output artifacts/dual_market/oos_split_injury_v21.csv
```

## 3. Run week-safe market comparison

```bash
python -m score_model.compare_split_market_v21 \
  --market-csv data/historical_market.csv \
  --qb-predictions artifacts/dual_market/oos_qb_common_v21.csv \
  --split-predictions artifacts/dual_market/oos_split_injury_v21.csv \
  --output artifacts/dual_market/split_injury_market_v21.json
```

## 4. View result

```bash
python -m json.tool artifacts/dual_market/split_injury_market_v21.json
```

Interpretation:
- positive ROI delta = injury split improved ROI
- negative Brier delta = injury split improved probability quality

Promote the split injury configuration only if it improves enough of these
metrics to justify the additional complexity.
