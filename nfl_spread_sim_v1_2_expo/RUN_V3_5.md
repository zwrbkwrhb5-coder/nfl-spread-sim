# v3.5 Anchor Robustness

v3.4 showed:

- spread anchor beats the standalone model but not the sportsbook market,
- total anchor very slightly beats both the standalone model and the market.

The total improvement is small enough that it could be noise.

v3.5 makes the test harder.

## Frozen-season design

For each target season:

### 2023
Learn anchor weights from 2022 only.

### 2024
Learn anchor weights from 2022-2023 only.

### 2025
Learn anchor weights from 2022-2024 only.

The weights are frozen for the entire target season. No result from the target
season can alter that season's weight.

## Paired week-block bootstrap

v3.5 resamples entire NFL weeks and calculates paired error differences:

```text
anchor MAE - market MAE
anchor RMSE - market RMSE
anchor MAE - standalone model MAE
anchor RMSE - standalone model RMSE
```

Negative values favor the anchor.

The important columns are:

```text
point_delta
ci_low_95
ci_high_95
probability_anchor_better
```

If the anchor-vs-market confidence interval spans zero, the apparent advantage
is not robust enough to call a demonstrated improvement.

## Exploratory 5+ total-gap check

v3.4 made the 5+ total-gap region look interesting.

Because that threshold was noticed AFTER looking at the data, v3.5 labels the
5+ analysis EXPLORATORY. Do not use it as a production betting filter.

## Run

```bash
unzip -n nfl_spread_totals_v3_5_anchor_robustness.zip
source .venv-model/bin/activate

python -m pytest v35_tests -q

python -m score_model.anchor_robustness_v35 \
  --predictions artifacts/dual_market/oos_qb_raw_context_v28.csv \
  --market-csv data/historical_market.csv \
  --output-dir artifacts/anchor_robustness_v35 \
  --first-eval-season 2023 \
  --min-prior-rows 200 \
  --bootstrap-reps 5000
```

Then:

```bash
cat artifacts/anchor_robustness_v35/frozen_weights.csv
cat artifacts/anchor_robustness_v35/overall_metrics.csv
cat artifacts/anchor_robustness_v35/by_season.csv
cat artifacts/anchor_robustness_v35/paired_week_bootstrap.csv
cat artifacts/anchor_robustness_v35/exploratory_gap_5plus.csv
```

## Decision rule

Do not change the v3.1 live model yet.

Promote the total market anchor only if the frozen-season results remain
competitive and the paired week-block uncertainty supports the improvement.

Spread should remain unpromoted unless it demonstrates incremental signal over
the market in a harder validation.
