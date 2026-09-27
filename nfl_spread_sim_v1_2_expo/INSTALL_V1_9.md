# v1.9 Injury / Availability Layer

This adds a historical injury layer on top of the QB-enhanced model.

## Features

Spread:
- relative total injury burden
- QB injury burden
- OL injury burden
- secondary injury burden
- skill-player injury burden

Totals:
- combined game injury burden for the same groups

The default historical evaluation stops at 2024 because this release does not
assume reliable public nflverse injury coverage beyond that season.

## 1. Install + test

```bash
unzip -n nfl_spread_totals_v1_9_injury_layer.zip
source .venv-model/bin/activate
python -m pytest injury_tests/test_injury_layer_v19.py -q
```

## 2. Build injury-enhanced training data

```bash
python -m score_model.injury_layer_v19 \
  --training-csv artifacts/dual_market/training_games_qb_v17.csv \
  --start-season 2018 \
  --end-season 2024 \
  --output artifacts/dual_market/training_games_qb_injury_v19.csv
```

You may also supply your own normalized/raw injury CSV with:

```bash
--injury-csv path/to/injuries.csv
```

## 3. Run common-sample injury ablation

```bash
python -m score_model.injury_ablation_v19 \
  --training-csv artifacts/dual_market/training_games_qb_injury_v19.csv \
  --first-test-season 2022 \
  --max-test-season 2024 \
  --output artifacts/dual_market/injury_ablation_v19.json
```

## 4. View

```bash
python -m json.tool artifacts/dual_market/injury_ablation_v19.json
```

Negative delta MAE/RMSE means the injury layer improved the QB baseline.
