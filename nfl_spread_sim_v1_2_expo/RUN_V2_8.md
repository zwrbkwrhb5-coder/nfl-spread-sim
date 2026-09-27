# v2.8 Leakage-Safe QB + Live Consistency

v2.8 fixes two important modeling problems before more live betting:

1. Historical QB shrinkage can no longer use future league averages.
2. Live QB/team transformations now match the historical training pipeline.

It also fixes:
- live opponent-adjusted EPA/success mismatch
- historical raw QB vs live shrunk QB mismatch
- duplicate opposite-side spread recommendations
- silent use of unknown outdoor weather for totals

## Architecture

### Spread
- EPA / success differential
- leakage-safe QB shrinkage (100 prior dropbacks)
- v2.6 probability calibration
- v2.7 spread selection

### Totals
- totals-specific team features
- true opponent-adjusted features
- leakage-safe QB shrinkage
- context/weather
- v2.6 probability calibration / selection

---

## 1. Install and test

```bash
unzip -n nfl_spread_totals_v2_8_leakage_safe_live.zip
source .venv-model/bin/activate
python -m pytest v28_tests -q
```

---

## 2. Build the leakage-safe historical table

```bash
python -m score_model.qb_safe_v28 \
  --training-csv artifacts/dual_market/training_games_qb_context_v24.csv \
  --output artifacts/dual_market/training_games_qb_safe_context_v28.csv \
  --shrinkage-dropbacks 100
```

Expected:
- `Future/same-week target rows: 0`
- `Past-only QB shrinkage: true`

---

## 3. Exact common-sample OOS comparison

This compares the current raw-QB architecture against leakage-safe QB
shrinkage on the exact same games.

```bash
python -m score_model.qb_safe_oos_v28 \
  --training-csv artifacts/dual_market/training_games_qb_safe_context_v28.csv \
  --first-test-season 2022 \
  --raw-output artifacts/dual_market/oos_qb_raw_context_v28.csv \
  --safe-output artifacts/dual_market/oos_qb_safe_context_v28.csv \
  --report-output artifacts/dual_market/qb_safe_oos_v28.json
```

View:

```bash
python -m json.tool artifacts/dual_market/qb_safe_oos_v28.json
```

Negative safe-minus-raw MAE/RMSE deltas are improvements.

---

## 4. Rebuild calibration using ONLY the clean v2.8 OOS predictions

Do not reuse the old v2.6 calibration results.

```bash
python -m score_model.calibration_selection_v26 \
  --market-csv data/historical_market.csv \
  --predictions artifacts/dual_market/oos_qb_safe_context_v28.csv \
  --output-dir artifacts/calibration_v28
```

View:

```bash
python -m json.tool artifacts/calibration_v28/report.json
cat artifacts/calibration_v28/summary_by_season.csv
```

---

## 5. Rebuild v2.7 spread stability using the clean calibration

```bash
python -m score_model.spread_stability_v27 \
  --calibrated-sides artifacts/calibration_v28/calibrated_sides.csv \
  --v26-selected-rows artifacts/calibration_v28/selected_rows.csv \
  --output-dir artifacts/spread_stability_v28 \
  --min-bets 60
```

View:

```bash
python -m json.tool artifacts/spread_stability_v28/report.json
cat artifacts/spread_stability_v28/summary_by_season.csv
```

---

## 6. Export live parameters

Totals:

```bash
python -m score_model.latest_calibration_params_v26 \
  --selected-rows artifacts/calibration_v28/selected_rows.csv \
  --output artifacts/calibration_v28/latest_params.json
```

Spreads:

```bash
python -m score_model.latest_spread_stability_params_v27 \
  --calibrated-sides artifacts/calibration_v28/calibrated_sides.csv \
  --output artifacts/spread_stability_v28/latest_params.json \
  --min-bets 60
```

---

## 7. Historical team artifact required for consistent live totals

v2.8 intentionally refuses to fake opponent-adjusted EPA.

The normal v0.2 training pipeline saved:

```text
artifacts/team_games.parquet
```

Check:

```bash
ls -lh artifacts/team_games.parquet
```

If it is missing, rebuild it with the original training pipeline before
running the live v2.8 model.

---

## 8. Live market file

Your existing market CSV can still be used. v2.8 also supports optional
columns:

```text
home_qb_id
home_qb_name
away_qb_id
away_qb_name
temp
wind
```

QB overrides are useful when the most recent starter is not the expected
starter for the upcoming game.

For outdoor games, v2.8 requires temp + wind before ranking the total.
It will still produce a spread projection without weather.

---

## 9. Run the consistent live model

Example:

```bash
python -m score_model.live_consistent_v28 \
  --training-csv artifacts/dual_market/training_games_qb_safe_context_v28.csv \
  --oos-csv artifacts/dual_market/oos_qb_safe_context_v28.csv \
  --team-games artifacts/team_games.parquet \
  --qb-games artifacts/dual_market/qb_games_v17.csv \
  --market-csv live_data/week3_2026_market.csv \
  --calibrated-sides artifacts/calibration_v28/calibrated_sides.csv \
  --total-params artifacts/calibration_v28/latest_params.json \
  --spread-params artifacts/spread_stability_v28/latest_params.json \
  --season 2026 \
  --week 3 \
  --output artifacts/live/week3_2026_ranked_v28.csv
```

v2.8 keeps only ONE best side per game/market, so a Top 5 list can no
longer contain both teams from the same spread.

---

## Promotion gate

Do not judge v2.8 by whether it makes the old live bets look better.

Promote it if:
1. the leakage audit stays clean,
2. safe-QB OOS accuracy is competitive with or better than raw QB,
3. recalibrated market results remain stable,
4. live features are complete and consistent.

The earlier Miami and Dallas bets remain v2.3 forward-test observations.
They should not be relabeled as v2.8 picks after the fact.
