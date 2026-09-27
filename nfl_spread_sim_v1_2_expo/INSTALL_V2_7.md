# v2.7 Spread Stability Filter

v2.6 improved spreads overall but broke down badly in 2025.
v2.7 is designed specifically to reduce that instability.

It does NOT change the football spread projection.
It changes only probability shrinkage and bet selection.

## What v2.7 tests

Probability shrinkage:
- 10%
- 25%
- 40%
- 55%
- 70%

Minimum calibrated edges:
- 0%
- 0.5%
- 1%
- 1.5%
- 2%
- 3%
- 5%

Rolling historical windows:
- 26 weeks
- 52 weeks
- 78 weeks
- 104 weeks
- full prior history

## Stability logic

For each historical week, using ONLY earlier weeks:

1. Test candidate shrinkage/threshold/window configurations.
2. Require a minimum historical bet count.
3. Score conservative ROI using a lower-confidence estimate.
4. Penalize season-to-season ROI volatility.
5. Penalize a badly negative worst historical season.
6. Apply the selected configuration to the current week.

No same-week or future outcomes are used.

## 1. Install + test

```bash
unzip -n nfl_spread_totals_v2_7_spread_stability.zip
source .venv-model/bin/activate
python -m pytest stability_tests/test_spread_stability_v27.py -q
```

## 2. Run spread stability validation

This uses the v2.6 outputs you already generated:

```bash
python -m score_model.spread_stability_v27 \
  --calibrated-sides artifacts/calibration_v26/calibrated_sides.csv \
  --v26-selected-rows artifacts/calibration_v26/selected_rows.csv \
  --output-dir artifacts/spread_stability_v27 \
  --min-bets 60
```

## 3. View overall report

```bash
python -m json.tool artifacts/spread_stability_v27/report.json
```

## 4. View season stability

```bash
cat artifacts/spread_stability_v27/summary_by_season.csv
```

The key question is whether 2025 improves without destroying 2022-2024.

## 5. Produce current spread-selection parameters

Only after reviewing the historical result:

```bash
python -m score_model.latest_spread_stability_params_v27 \
  --calibrated-sides artifacts/calibration_v26/calibrated_sides.csv \
  --output artifacts/spread_stability_v27/latest_params.json \
  --min-bets 60
```

Then:

```bash
python -m json.tool artifacts/spread_stability_v27/latest_params.json
```

## Promotion rule

Promote v2.7 only if it improves spread season stability and does not obtain
that improvement merely by collapsing the number of bets to a tiny sample.

Totals remain on the v2.6 calibrated QB + context/weather architecture.
