# NFL SIM v4.5 — Best Odds Shopping

This adds an odds-shopping view for every app-selected play.

## What changes

For a model pick, the app now shows:

- selected line
- selected American odds
- sportsbook
- model edge
- up to 8 alternative book quotes for the same pick
- the app-selected best quote highlighted with ★

The ranking uses the model's calculated edge, so it evaluates **line + juice together**.
It does not simply choose the highest-looking American odds.

Example:

```text
UNDER 50.5

★ DraftKings   50.5  -110    +2.62%
  FanDuel      50.5  -115    +1.55%
  BetMGM       51.0  -120    +1.22%
```

## Install

Download this ZIP from ChatGPT, then upload it to:

```text
/workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo/
```

Then:

```bash
cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo
unzip -o nfl_app_v4_5_best_odds_shop.zip
```

## Export the odds board

Use the all-candidates v3.1 file as well as the best file:

```bash
source .venv-model/bin/activate

python -m score_model.export_mobile_v45 \
  --best artifacts/live/week3_2026_best_v31.csv \
  --all-candidates artifacts/live/week3_2026_all_candidates_v31.csv \
  --ledger artifacts/live/forward_test_ledger_v39.csv \
  --results artifacts/live/week3_2026_results.csv \
  --shadow artifacts/live/week3_2026_totals_shadow_v38.csv \
  --historical artifacts/market_stack_v37/overall_metrics.csv \
  --season 2026 \
  --week 3 \
  --output mobile/generatedData.ts
```

If your actual all-candidates filename is different, replace only that path.

## Start Expo

```bash
cd mobile
npx expo start --tunnel -c
```
