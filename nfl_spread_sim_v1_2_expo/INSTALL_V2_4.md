# v2.4 Context + Weather Layer

Adds schedule/environment context to the QB-enhanced baseline.

Spread candidates:
- rest differential
- short-week differential
- long-rest differential
- divisional game
- indoor
- grass

Totals candidates:
- divisional game
- indoor
- grass
- cold degrees
- wind above 10 mph
- cold-game flag
- windy-game flag

## 1. Install + test

```bash
unzip -n nfl_spread_totals_v2_4_context_weather.zip
source .venv-model/bin/activate
python -m pytest context_tests/test_context_v24.py -q
```

## 2. Build context-enhanced historical table

```bash
python -m score_model.context_weather_v24 \
  --training-csv artifacts/dual_market/training_games_qb_v17.csv \
  --output artifacts/dual_market/training_games_qb_context_v24.csv
```

## 3. Run common-sample ablation

```bash
python -m score_model.context_ablation_v24 \
  --training-csv artifacts/dual_market/training_games_qb_context_v24.csv \
  --first-test-season 2022 \
  --output artifacts/dual_market/context_ablation_v24.json
```

## 4. View

```bash
python -m json.tool artifacts/dual_market/context_ablation_v24.json
```

Negative delta MAE/RMSE means the context layer improved the QB baseline.
