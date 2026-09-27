# Model v0.4 — QB Integration + A/B Backtest

## Goal

Measure whether the QB layer actually improves the model.

## Comparison

### Model A
Team-only:
- recency-weighted team metrics
- opponent-adjusted EPA
- opponent-adjusted success rate

### Model B
Team + QB:
all Model A features plus:
- QB EPA/dropback
- QB success rate
- QB CPOE
- QB sack rate
- QB explosive-pass rate
- QB interception rate

## Testing method

Walk-forward by season.

Example:
- train through 2021, test 2022
- train through 2022, test 2023
- train through 2023, test 2024
- train through 2024, test 2025

The script compares both models on the SAME games where QB data is available.

## Interpretation

`delta_mae_vs_common_team`

- negative = QB model improved MAE
- positive = QB model got worse

`delta_rmse_vs_common_team`

- negative = QB model improved RMSE
- positive = QB model got worse

We do not keep a feature because it sounds smart.
We keep it only if the backtest shows value or if it improves calibration later.

## Run

```bash
python -m src.compare_models \
  --start-season 2018 \
  --end-season 2025 \
  --first-test-season 2022
```

Outputs:

- `artifacts/model_comparison.json`
- `artifacts/team_only_walk_forward.csv`
- `artifacts/team_qb_walk_forward.csv`
- `artifacts/spread_model_v04.joblib`
