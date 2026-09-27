# v3.4 Market-Anchored Model

Calibration diagnostics showed diminishing returns. v3.4 moves upstream and
tests whether the football model contains useful information AFTER accounting
for the sportsbook market.

## Core idea

Suppose:

```text
sportsbook implied home margin = +3
football model predicts home margin = +7
```

The model-market disagreement is:

```text
+4 points
```

Instead of assuming all 4 points are signal, v3.4 asks historical prior games:

> When this model disagreed with the market, how much of that disagreement
> actually showed up in the final result?

It learns:

```text
actual_market_residual ~= anchor_weight * model_market_gap
```

strictly from prior weeks.

Examples:

```text
anchor weight = 0.00 -> trust market
anchor weight = 0.20 -> keep 20% of our disagreement
anchor weight = 0.50 -> split the difference
anchor weight = 1.00 -> trust raw model
```

The learned weight is clipped to `[0, 1]` so v3.4 cannot extrapolate beyond
both forecasts.

## Why this matters

Our raw probabilities were overconfident and calibration had to crush them.

A market anchor attacks that problem BEFORE probability calibration by making
the underlying point projection more realistic.

## Evaluation

v3.4 begins evaluation in 2023.

For every week in 2023-2025:

1. use only rows with an earlier `week_key`,
2. learn spread and total anchor weights,
3. freeze those weights for the current week,
4. create the anchored point projection,
5. compare:
   - sportsbook market
   - raw v2.8/v3.1 football model
   - anchored model

This test does NOT optimize ROI.

## Run

```bash
unzip -n nfl_spread_totals_v3_4_market_anchor.zip
source .venv-model/bin/activate

python -m pytest v34_tests -q

python -m score_model.market_anchor_v34 \
  --predictions artifacts/dual_market/oos_qb_raw_context_v28.csv \
  --market-csv data/historical_market.csv \
  --output-dir artifacts/market_anchor_v34 \
  --min-prior-rows 200 \
  --first-eval-season 2023
```

Then print:

```bash
cat artifacts/market_anchor_v34/overall_metrics.csv
cat artifacts/market_anchor_v34/by_season.csv
cat artifacts/market_anchor_v34/gap_buckets.csv
```

## Promotion rule

Do not promote the anchor because it produces more bets.

First ask:

1. Does anchored MAE beat the raw model?
2. Does it beat or at least approach the sportsbook market?
3. Does RMSE improve?
4. Are improvements present across multiple seasons?
5. Is the learned weight stable and sensible?

If yes, the next step is to rebuild probability calibration on top of the
anchored point forecast.

If no, keep the standalone model and focus on new predictive features.
