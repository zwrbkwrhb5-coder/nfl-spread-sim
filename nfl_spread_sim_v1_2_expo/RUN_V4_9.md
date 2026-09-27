# NFL SIM v4.9 — Moneyline Market

This patch adds **moneyline** support to the mobile app.

## What changed

- Adds a third market: **moneyline**
- Shows moneyline on:
  - Dashboard
  - Game detail screen
  - Best Bets screen
  - My Tracking
  - Performance
- Lets you log a moneyline bet with **I BET THIS**
- Settles tracked moneyline bets correctly from final game results
- Adds `score_model/export_mobile_v49.py` so the mobile export can include moneyline rows

## Important

The UI will only show moneyline picks if your exported mobile data actually contains `market = moneyline` rows.
If your source CSVs do not have moneyline candidates yet, the app will still work — it just will not display moneyline cards until that data exists.

## Install

```bash
cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo
unzip -o nfl_app_v4_9_moneyline_market.zip
```

## Re-export mobile data

```bash
python -m score_model.export_mobile_v49   --best artifacts/live/week3_2026_best_v31.csv   --all-candidates artifacts/live/week3_2026_all_candidates_v31.csv   --ledger artifacts/live/forward_test_ledger_v39.csv   --results artifacts/live/week3_2026_results.csv   --shadow artifacts/live/week3_2026_totals_shadow_v38.csv   --historical artifacts/market_stack_v37/overall_metrics.csv   --season 2026   --week 3   --output mobile/generatedData.ts
```

## Run the app

```bash
cd mobile
npx expo start --tunnel -c
```
