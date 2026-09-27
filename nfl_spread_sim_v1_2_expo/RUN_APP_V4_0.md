# NFL Model App v4.0 — Real Results Dashboard

This replaces the demo-only mobile data with a generated snapshot from the
actual model artifacts already in the project.

No new npm packages are required.

## New screens

- Games
- Game Analysis
- Qualified Picks
- Performance / CLV

## Data sources

The exporter reads:

```text
artifacts/live/week3_2026_best_v31.csv
artifacts/live/forward_test_ledger_v39.csv
artifacts/live/week3_2026_totals_shadow_v38.csv
artifacts/market_stack_v37/overall_metrics.csv
```

and generates:

```text
mobile/generatedData.ts
```

The Expo app imports that generated file directly.

## Install

From the repository root:

```bash
unzip -o nfl_app_v4_0_real_results_dashboard.zip
source .venv-model/bin/activate
```

## Export current model data into the app

```bash
python -m score_model.export_mobile_v40 \
  --best artifacts/live/week3_2026_best_v31.csv \
  --ledger artifacts/live/forward_test_ledger_v39.csv \
  --shadow artifacts/live/week3_2026_totals_shadow_v38.csv \
  --historical artifacts/market_stack_v37/overall_metrics.csv \
  --season 2026 \
  --week 3 \
  --output mobile/generatedData.ts
```

## Run Expo

```bash
cd mobile
npm install
npx expo start -c
```

Scan the QR code with Expo Go.

## Refresh workflow

Whenever v3.1 / v3.8 / v3.9 changes:

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

Then reload the Expo app.

## Status colors

```text
Green  = production qualifier
Gray   = no bet
Purple = shadow / research
Blue   = forward-test tracking
```

## Important

The app does not promote v3.8 shadow totals into production.

The Performance screen also treats historical MAE as diagnostic evidence, not
as proof of profitability.
