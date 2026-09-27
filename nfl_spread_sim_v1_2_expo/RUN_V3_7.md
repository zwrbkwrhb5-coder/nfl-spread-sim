# v3.7 Frozen Market Stack

v3.6 rejected the post-hoc 5+ point total-gap rule on the 2022 holdout.

Rather than search more thresholds, v3.7 tests a low-dimensional market-aware
forecast.

## Four forecasts are compared

### 1. Raw sportsbook market
Spread:
`market_margin = -home_spread`

Total:
`total_line`

### 2. Historically calibrated market only
Prior seasons fit:

Spread:
`actual_margin ~ market_margin`

Total:
`actual_total ~ total_line`

This is important. If the market has a small historical slope/intercept bias,
we should not mistakenly credit our football model for correcting it.

### 3. Standalone football model
`pred_margin` and `pred_total`

### 4. Market + football-model stack
Prior seasons fit:

Spread:
`actual_margin ~ market_margin + pred_margin`

Total:
`actual_total ~ total_line + pred_total`

Only two predictors plus an intercept are used. There is no threshold mining
and no ROI optimization.

## Frozen validation

For 2023:
train on 2022 only.

For 2024:
train on 2022-2023 only.

For 2025:
train on 2022-2024 only.

The coefficients are frozen for each target season.

## Critical comparison

The most important test is:

`stack vs market_calibrated`

If the stack beats raw market but NOT market-only calibration, then our
football model is not providing incremental information.

If the stack beats market-only calibration consistently, the football model
has evidence of incremental predictive value beyond the line itself.

## Run

```bash
unzip -n nfl_spread_totals_v3_7_market_stack.zip
source .venv-model/bin/activate

python -m pytest v37_tests -q

python -m score_model.market_stack_v37 \
  --predictions artifacts/dual_market/oos_qb_raw_context_v28.csv \
  --market-csv data/historical_market.csv \
  --output-dir artifacts/market_stack_v37 \
  --first-eval-season 2023 \
  --min-prior-rows 200 \
  --bootstrap-reps 5000
```

Then:

```bash
cat artifacts/market_stack_v37/coefficients.csv
cat artifacts/market_stack_v37/overall_metrics.csv
cat artifacts/market_stack_v37/by_season.csv
cat artifacts/market_stack_v37/paired_week_bootstrap.csv
```

## Promotion criteria

Do not alter the v3.1 live betting model from this test alone.

A market stack becomes a serious candidate only if:

- coefficients are reasonably stable,
- stack improves on standalone model,
- stack improves on raw market,
- most importantly, stack improves on market-only calibration,
- improvement appears across seasons,
- paired week-block uncertainty supports the improvement.

If not, stop tuning market blends and return to genuinely new football
features.
