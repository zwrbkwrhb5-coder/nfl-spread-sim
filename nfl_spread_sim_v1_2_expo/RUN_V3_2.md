# v3.2 Calibration Diagnostics

This is a diagnostic build. It does NOT replace the live calibration layer
automatically.

## Question

The v3.1 live output showed raw probabilities such as 0.63-0.67 collapsing
into a narrow calibrated range around 0.51-0.53.

v3.2 tests whether that compression is useful calibration or excessive
flattening.

## Methods

### `raw`
The original model probability.

### `isotonic_v26`
The already-deployed, strict walk-forward isotonic probability stored in
`artifacts/calibration_v28_raw/calibrated_sides.csv`.

### `platt_v32`
A new strict walk-forward logistic/Platt recalibration.

For week W it is trained only on settled rows where:

```text
week_key < W
```

It fits:

```text
win ~ logistic(logit(raw_probability))
```

No current-week result enters the fit.

## Important experimental design

The downstream live decision rule is frozen so we isolate CALIBRATION rather
than retuning everything at the same time.

Spread:

```text
probability shrinkage = 0.40
minimum edge = 0.01
```

Totals:

```text
probability shrinkage = 0.25
minimum edge = 0.01
```

For each method, exactly one side is kept per game/market: the side with the
highest post-shrinkage edge.

## Outputs

- `calibration_metrics.csv`
  - Brier
  - log loss
  - 10-bin ECE
  - probability dispersion/resolution

- `reliability_bins.csv`
  - predicted vs actual by probability bucket

- `high_raw_probability_compression.csv`
  - what happens to raw 55%, 60%, 65%, and 70%+ signals

- `isotonic_compression_summary.csv`
  - how aggressively deployed isotonic moves probabilities toward 50%

- `fixed_rule_betting_summary.csv`
  - bets / win rate / units / ROI using CURRENT live rules

- `fixed_rule_by_season.csv`
  - season stability

- `week_block_bootstrap_roi.csv`
  - descriptive 95% ROI interval from resampling whole NFL weeks

- `fixed_rule_selected_rows.csv`
  - the exact selected side for every game/market/method

## Run

```bash
unzip -n nfl_spread_totals_v3_2_calibration_diagnostics.zip
source .venv-model/bin/activate

python -m pytest v32_tests -q

python -m score_model.calibration_diagnostics_v32 \
  --calibrated-sides artifacts/calibration_v28_raw/calibrated_sides.csv \
  --output-dir artifacts/calibration_v32 \
  --bootstrap-reps 2000
```

Then print the important outputs:

```bash
cat artifacts/calibration_v32/calibration_metrics.csv
cat artifacts/calibration_v32/isotonic_compression_summary.csv
cat artifacts/calibration_v32/fixed_rule_betting_summary.csv
cat artifacts/calibration_v32/fixed_rule_by_season.csv
cat artifacts/calibration_v32/week_block_bootstrap_roi.csv
```

## Promotion rule

Do not promote Platt just because it has the highest historical ROI.

A calibration change should preferably show:

1. lower or comparable Brier score,
2. lower or comparable log loss,
3. sensible reliability bins,
4. adequate probability resolution,
5. no severe season-specific breakdown,
6. better or comparable betting results under the SAME fixed decision rules,
7. uncertainty that does not make the apparent improvement obviously fragile.

If results are mixed, keep v3.1 isotonic until we run a stronger nested/frozen
validation.
