# v2.9 Raw-QB Consistent Live Runner

v2.9 is the cleaned live architecture selected by the v2.8 validation.

## Locked architecture

### Spread
- EPA / success differential
- RAW pregame QB form
- v2.6 isotonic probability calibration
- v2.7 spread stability selection

Current raw-QB spread selector found by the historical pipeline:

- probability shrinkage lambda: `0.40`
- minimum edge: `0.01`
- rolling window: `78 weeks`

The runner reads those values from:
`artifacts/spread_stability_v28_raw/latest_params.json`

### Totals
- totals-specific team features
- RAW pregame QB form
- opponent-adjusted EPA/success
- context/weather
- v2.6 calibration + selection

The runner reads total parameters from:
`artifacts/calibration_v28_raw/latest_params.json`

---

## 1. Install and test

```bash
unzip -n nfl_spread_totals_v2_9_raw_qb_live.zip
source .venv-model/bin/activate
python -m pytest v29_tests -q
```

---

## 2. Make sure BOTH clean parameter files exist

Spread:

```bash
python -m score_model.latest_spread_stability_params_v27 \
  --calibrated-sides artifacts/calibration_v28_raw/calibrated_sides.csv \
  --output artifacts/spread_stability_v28_raw/latest_params.json \
  --min-bets 60
```

Totals:

```bash
python -m score_model.latest_calibration_params_v26 \
  --selected-rows artifacts/calibration_v28_raw/selected_rows.csv \
  --output artifacts/calibration_v28_raw/latest_params.json
```

Check:

```bash
python -m json.tool artifacts/spread_stability_v28_raw/latest_params.json
python -m json.tool artifacts/calibration_v28_raw/latest_params.json
```

---

## 3. Historical raw team-state artifact

v2.9 needs the historical raw team-game artifact so live opponent adjustment
uses the same code as historical training.

Check:

```bash
ls -lh artifacts/team_games.parquet
```

If it is missing, rebuild the original team feature pipeline before running
v2.9. The runner intentionally does NOT replace adjusted EPA with raw EPA.

---

## 4. Current market CSV

Your existing `live_data/week3_2026_market.csv` can be used.

Required columns:

```text
game_id
season
week
away_team
home_team
home_spread
total_line
home_spread_odds
away_spread_odds
over_odds
under_odds
```

Optional but recommended:

```text
home_qb_id
home_qb_name
away_qb_id
away_qb_name
temp
wind
```

QB overrides matter whenever the expected starter is different from the
team's most recent primary QB.

For outdoor games, totals require actual/current temp and wind. Missing
outdoor weather excludes that total rather than silently using fake weather.

---

## 5. Run v2.9

```bash
python -m score_model.live_raw_qb_v29 \
  --training-csv artifacts/dual_market/training_games_qb_context_v24.csv \
  --oos-csv artifacts/dual_market/oos_qb_raw_context_v28.csv \
  --team-games artifacts/team_games.parquet \
  --qb-games artifacts/dual_market/qb_games_v17.csv \
  --market-csv live_data/week3_2026_market.csv \
  --calibrated-sides artifacts/calibration_v28_raw/calibrated_sides.csv \
  --total-params artifacts/calibration_v28_raw/latest_params.json \
  --spread-params artifacts/spread_stability_v28_raw/latest_params.json \
  --season 2026 \
  --week 3 \
  --output artifacts/live/week3_2026_ranked_v29.csv
```

---

## What v2.9 fixes

- no 100-dropback QB shrinkage
- no future league-average QB leakage
- raw historical/live QB form has matching semantics
- true live opponent-adjusted team features
- context/weather remains totals-only
- no outdoor total without temp + wind
- exactly one side per game/market
- calibrated probability is shown separately from raw probability
- live selector reads the clean raw-QB historical parameters

---

## Important forward-testing rule

Miami +10.5 (-115) and Dallas +3.5 (-112) remain v2.3-era forward-test bets.

Do not rewrite them as v2.9 picks after seeing the new model output.

New qualifying picks after this runner is working should be logged as v2.9.
