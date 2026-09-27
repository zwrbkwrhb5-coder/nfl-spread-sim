# v3.8 Closing-Line Audit + Totals Shadow Model

The historical market file contains both:

```text
home_spread
total_line
```

and explicit:

```text
closing_home_spread
closing_total
```

The sample inspected before v3.8 showed those values matching.

v3.8 verifies the full file instead of assuming.

## Why timing matters

v3.7's promising total stack was trained/evaluated with historical market
information.

If `total_line` is effectively a closing line, then using a Sunday-morning
current line as though it were the same feature creates a timing/distribution
mismatch.

v3.8 therefore DOES NOT promote the stack into the production betting model.

It creates a shadow-only 2026 model.

## Step 1 — install and test

```bash
unzip -n nfl_spread_totals_v3_8_closing_line_shadow.zip
source .venv-model/bin/activate

python -m pytest v38_tests -q
```

## Step 2 — historical timing audit

```bash
python -m score_model.historical_market_timing_v38 \
  --market-csv data/historical_market.csv \
  --output-dir artifacts/market_timing_v38
```

## Step 3 — run a fresh multi-book pull + totals shadow

Your `THE_ODDS_API_KEY` must already be exported.

```bash
python -m score_model.run_shadow_v38 \
  --season 2026 \
  --week 3 \
  --max-age-minutes 30 \
  --snapshot-label current
```

This first runs v3.1 to refresh all sportsbook quotes.

Then it:

1. takes fresh totals quotes,
2. collapses over/under duplicates to one total per book,
3. computes the median current total across books,
4. fits ONE frozen 2026 stack on all historical 2022-2025 OOS rows,
5. computes a current-line shadow projection,
6. logs the snapshot.

## Important terminology

If the historical timing audit says:

```text
closing_alias
```

then:

```text
shadow_stack_total_current_proxy
```

is NOT a timing-matched production forecast.

It means:

> what the closing-line-trained stack would output if today's current
> multi-book median were substituted for the closing market feature.

That is useful for forward tracking, not for retroactively claiming an edge.

## Outputs

Historical timing:

```text
artifacts/market_timing_v38/timing_audit.csv
artifacts/market_timing_v38/timing_classification.json
```

Current shadow:

```text
artifacts/live/week3_2026_totals_shadow_v38.csv
```

Persistent history:

```text
artifacts/live/totals_shadow_history_v38.csv
```

Frozen 2026 coefficients:

```text
artifacts/live/totals_stack_coefficients_2026_v38.json
```

## Snapshot labels

Use labels such as:

```text
current
pregame_60m
pregame_15m
closing_candidate
```

Example near kickoff:

```bash
python -m score_model.run_shadow_v38 \
  --season 2026 \
  --week 3 \
  --max-age-minutes 15 \
  --snapshot-label closing_candidate
```

Do not call a snapshot a true close unless it was actually collected at the
appropriate market-close time.

## Production status

Spread:
- v3.1 unchanged.

Totals production:
- v3.1 unchanged.

Totals market-stack:
- v3.8 shadow only.

5+ point gap rule:
- rejected by v3.6 holdout.

No threshold should be loosened to generate action.
