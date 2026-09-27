# v2.2 Opening-Drive Tendencies

This release adds pregame-only opening-drive tendency features to the QB baseline.

Features include:
- opening-drive points
- EPA
- success rate
- turnover rate
- scoring rate
- TD rate
- defensive opening-drive allowed tendencies
- offense-vs-defense opening-drive matchup interactions

No current-game opening-drive result is used as a pregame feature.

## 1. Install + test

```bash
unzip -n nfl_spread_totals_v2_2_opening_drive.zip
source .venv-model/bin/activate
python -m pytest opening_drive_tests/test_opening_drive_v22.py -q
```

## 2. Build opening-drive-enhanced training data

```bash
python -m score_model.opening_drive_v22 \
  --training-csv artifacts/dual_market/training_games_qb_v17.csv \
  --start-season 2018 \
  --end-season 2025 \
  --halflife-games 6 \
  --min-games 3 \
  --output artifacts/dual_market/training_games_qb_opening_v22.csv
```

## 3. Run common-sample ablation

```bash
python -m score_model.opening_drive_ablation_v22 \
  --training-csv artifacts/dual_market/training_games_qb_opening_v22.csv \
  --first-test-season 2022 \
  --output artifacts/dual_market/opening_drive_ablation_v22.json
```

## 4. View

```bash
python -m json.tool artifacts/dual_market/opening_drive_ablation_v22.json
```

Negative `delta_mae` and `delta_rmse` mean opening-drive tendencies improved the QB baseline.
