# v3.3 Frozen Calibration Validation

v3.2 showed that Platt calibration has the best proper scoring metrics, but
the existing production probability shrinkage was originally selected for the
isotonic pipeline.

Applying that same shrinkage to Platt can double-shrink the probability.

v3.3 tests the methods more fairly.

## What is frozen

Minimum betting edge stays:

```text
1%
```

No ROI is used to tune the calibrator or probability shrinkage.

## How each future season is tested

For each market and calibration method:

1. take only PRIOR SEASONS,
2. test shrinkage lambda:
   - 0.25
   - 0.50
   - 0.75
   - 1.00
3. choose lambda with the lowest PRIOR-SEASON Brier score,
4. freeze that lambda for the entire next season,
5. select one side per game/market,
6. bet only when calibrated edge >= 1%,
7. move to the following season.

With data from 2022-2025:

- 2022 is calibration/training history,
- 2023 is selected using 2022 only,
- 2024 is selected using 2022-2023 only,
- 2025 is selected using 2022-2024 only.

This is substantially harder to overfit than choosing a new ROI-optimal rule
every week.

## Run

```bash
unzip -n nfl_spread_totals_v3_3_frozen_calibration_validation.zip
source .venv-model/bin/activate

python -m pytest v33_tests -q

python -m score_model.frozen_calibration_validation_v33 \
  --probabilities artifacts/calibration_v32/walk_forward_probabilities.csv \
  --output-dir artifacts/calibration_v33 \
  --bootstrap-reps 3000
```

Then:

```bash
cat artifacts/calibration_v33/chosen_lambdas_by_target_season.csv
cat artifacts/calibration_v33/frozen_betting_summary.csv
cat artifacts/calibration_v33/frozen_by_season.csv
cat artifacts/calibration_v33/frozen_calibration_by_season.csv
cat artifacts/calibration_v33/frozen_week_block_bootstrap_roi.csv
```

## Interpretation

The preferred production method should not be chosen from ROI alone.

Look for:

- strong future-season Brier score,
- sensible lambda stability,
- reasonable number of bets,
- no severe single-season collapse,
- week-block ROI uncertainty,
- consistency between probability quality and betting results.

If Platt only looks good after ROI tuning, reject it.

If Platt looks good after this Brier-selected, season-frozen test, it becomes a
much stronger production candidate.
