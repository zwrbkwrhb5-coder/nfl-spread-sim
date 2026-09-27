# v2.6 Probability Calibration + Time-Safe Bet Selection

This release targets the model's largest measured weakness:
claimed probability edges are much larger than realized betting performance.

It uses the strongest current architecture:

- Spread: QB-enhanced model
- Totals: QB + context/weather model

Recommended prediction source:
`artifacts/dual_market/oos_qb_total_context_v25.csv`

## What v2.6 adds

1. Prior-week-only residual probability estimates
2. Prior-week-only isotonic calibration
3. Probability shrinkage toward 50%
4. Time-safe minimum-edge selection
5. Separate spread and totals tuning
6. Season-by-season diagnostics

Candidate shrinkage:
- 25%
- 50%
- 75%
- 100%

Candidate minimum edges:
- 0%
- 1%
- 2%
- 3%
- 5%
- 7.5%
- 10%

At each historical week, the current parameters are chosen using only earlier
weeks. Same-week results and future weeks are excluded.

## 1. Install + test

```bash
unzip -n nfl_spread_totals_v2_6_calibration_selection.zip
source .venv-model/bin/activate
python -m pytest calibration_tests/test_calibration_v26.py -q
```

## 2. Run strict time-safe calibration / selection

```bash
python -m score_model.calibration_selection_v26 \
  --market-csv data/historical_market.csv \
  --predictions artifacts/dual_market/oos_qb_total_context_v25.csv \
  --output-dir artifacts/calibration_v26
```

## 3. View report

```bash
python -m json.tool artifacts/calibration_v26/report.json
```

Also inspect:

```bash
cat artifacts/calibration_v26/summary_by_season.csv
```

## 4. Produce parameters for the next live slate

Only do this after reviewing the historical report:

```bash
python -m score_model.latest_calibration_params_v26 \
  --selected-rows artifacts/calibration_v26/selected_rows.csv \
  --output artifacts/calibration_v26/latest_params.json
```

View:

```bash
python -m json.tool artifacts/calibration_v26/latest_params.json
```

## Interpretation

The historical comparison contains:

- `raw_no_threshold`
- `calibrated_selected`

For a useful calibration/selection layer, look for:

- improved Brier score
- improved ROI
- sensible bet count
- better season stability

Do not promote a threshold merely because one historical bucket had a positive
ROI. The time-safe selector must improve results without future information.
