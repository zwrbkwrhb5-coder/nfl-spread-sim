# v3.6 Fixed 5-Point Gap Holdout

v3.5 found an exploratory pattern:

```text
TOTAL | abs(model total - market total) >= 5
72 games
59.7% same direction
+4.34 average signed market residual
```

That pattern was discovered by looking at the 2023-2025 evaluation data.

We MUST NOT tune a new production filter on those same seasons.

## One-shot threshold

v3.6 freezes the threshold at:

```text
abs(model - market) >= 5.0 points
```

No alternate threshold is searched.

## Holdout

The primary check is 2022.

2022 was not part of the v3.4/v3.5 gap-bucket discovery output, so it serves as
a reverse-time holdout for this specific pattern.

This is still NOT equivalent to a true future test because the broader model
development process has already used historical data.

If the 2022 result supports the pattern, the correct next step is:

```text
register 5+ total gap as a 2026 SHADOW signal
```

Do not automatically turn it into a real-money production rule.

## What v3.6 reports

For spread and total separately:

- number of 5+ point disagreements
- win rate following model direction
- Wilson 95% win-rate interval
- same-direction rate
- average/median signed final-score residual
- units/ROI when historical price columns are available
- week-block bootstrap ROI interval

The spread result is a control because v3.5 did not show meaningful large-gap
spread signal.

## Run

```bash
unzip -n nfl_spread_totals_v3_6_gap_holdout.zip
source .venv-model/bin/activate

python -m pytest v36_tests -q

python -m score_model.gap_holdout_v36 \
  --predictions artifacts/dual_market/oos_qb_raw_context_v28.csv \
  --market-csv data/historical_market.csv \
  --holdout-season 2022 \
  --discovery-start 2023 \
  --discovery-end 2025 \
  --output-dir artifacts/gap_holdout_v36 \
  --bootstrap-reps 5000
```

Then:

```bash
cat artifacts/gap_holdout_v36/summary.csv
cat artifacts/gap_holdout_v36/week_block_bootstrap_roi.csv
```

## Decision

### If 2022 total 5+ signal fails
Drop the 5-point idea. Keep the global total anchor as a research candidate.

### If 2022 total 5+ signal survives
Do NOT declare it profitable.

Register it as a fixed 2026 shadow flag and collect:
- opening line
- bet-time line
- closing line
- side
- model/market gap
- final result
- CLV

No threshold changes during the forward-test period.
