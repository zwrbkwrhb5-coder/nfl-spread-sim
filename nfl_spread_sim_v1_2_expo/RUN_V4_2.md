# NFL App v4.2 — Visual Results UI

This patch assumes v4.0 is already installed.

It does not add npm dependencies.

## What changes

Home:
- simpler green QUALIFIED / gray PASS status
- large pick + book
- visual Model Chance vs Break-even bars
- plain-language edge explanation
- direct Model Picks and Results buttons

Model Picks:
- qualifiers only
- large pick cards
- probability bars
- edge callout

Results:
- big record and units
- average CLV
- positive CLV rate
- entry -> close visual for every tracked pick
- pending / win / loss / push badges
- historical model-vs-market error bars

Game Breakdown:
- projected score card
- spread and total shown separately
- easy Model Chance vs Break-even visuals
- research/shadow section clearly separated

## Install from repository root

```bash
unzip -o nfl_app_v4_2_visual_results.zip
```

No npm install is required if your v4.0 app already runs.

If Expo is already open, stop it and restart:

```bash
cd mobile
npx expo start --tunnel -c
```

## Refresh data later

Continue using the existing v4.0 exporter:

```bash
cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo
source .venv-model/bin/activate

python -m score_model.export_mobile_v40 \
  --best artifacts/live/week3_2026_best_v31.csv \
  --ledger artifacts/live/forward_test_ledger_v39.csv \
  --shadow artifacts/live/week3_2026_totals_shadow_v38.csv \
  --historical artifacts/market_stack_v37/overall_metrics.csv \
  --season 2026 \
  --week 3 \
  --output mobile/generatedData.ts
```
