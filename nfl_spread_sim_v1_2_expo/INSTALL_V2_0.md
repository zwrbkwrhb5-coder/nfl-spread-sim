# v2.0 Player-Value Injury Model

Compares:
1. QB baseline
2. QB + basic injury burden (v1.9)
3. QB + player-value injury burden

The player-value model adds starter/depth importance on top of injury severity
and position value.

## 1. Install + test

```bash
unzip -n nfl_spread_totals_v2_0_player_value_injuries.zip
source .venv-model/bin/activate
python -m pytest injury_tests/test_player_value_v20.py -q
```

## 2. Build player-value injury training table

```bash
python -m score_model.player_value_injury_v20 \
  --training-csv artifacts/dual_market/training_games_qb_injury_v19.csv \
  --start-season 2018 \
  --end-season 2024 \
  --output artifacts/dual_market/training_games_qb_injury_pv_v20.csv
```

## 3. Run common-sample comparison

```bash
python -m score_model.player_value_ablation_v20 \
  --training-csv artifacts/dual_market/training_games_qb_injury_pv_v20.csv \
  --first-test-season 2022 \
  --max-test-season 2024 \
  --output artifacts/dual_market/player_value_ablation_v20.json
```

## 4. View

```bash
python -m json.tool artifacts/dual_market/player_value_ablation_v20.json
```

Key fields:
- `pv_delta_vs_base_mae`
- `pv_delta_vs_basic_mae`
- `pv_delta_vs_base_rmse`
- `pv_delta_vs_basic_rmse`

Negative values mean player-value injuries improve the comparison model.
