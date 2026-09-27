# NFL SIM v4.4 — Premium UI

This implements the approved black / gold / green NFL SIM design.

No new npm packages are required.

## Screens

- Home / Dashboard
- Best Bets
- Results
- Performance
- Game Detail

Started games are hidden automatically from Home and Picks using kickoff times.

Results remain visible after games finish.

The app uses remote NFL team-logo images and your actual v3.1/v3.9 model data.

## Important

`/mnt/data` is ChatGPT's workspace, not your GitHub Codespace.

Download the ZIP from ChatGPT first, then upload it into this Codespace folder:

```text
/workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo/
```

## Install

```bash
cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo

unzip -o nfl_app_v4_4_premium_nfl_sim.zip
```

## Re-export real settled data

```bash
source .venv-model/bin/activate

python -m score_model.export_mobile_v44 \
  --best artifacts/live/week3_2026_best_v31.csv \
  --ledger artifacts/live/forward_test_ledger_v39.csv \
  --results artifacts/live/week3_2026_results.csv \
  --shadow artifacts/live/week3_2026_totals_shadow_v38.csv \
  --historical artifacts/market_stack_v37/overall_metrics.csv \
  --season 2026 \
  --week 3 \
  --output mobile/generatedData.ts
```

## Start Expo

```bash
cd mobile
npx expo start --tunnel -c
```

No `npm install` is necessary if the existing mobile app already runs.
