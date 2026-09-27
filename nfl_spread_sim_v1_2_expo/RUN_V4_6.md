# NFL SIM v4.6 — Top 2 Odds

This version shows the two best sportsbook options for every model-selected play.

## App display

Each selected play now highlights:

```text
TOP 2 AVAILABLE OPTIONS

★ #1 BEST
DraftKings
UNDER 50.5   -110
+2.62% edge

#2 NEXT BEST
FanDuel
UNDER 50.5   -115
+1.55% edge
```

The ranking uses model edge and therefore evaluates both the betting line and the juice.

## Install

Download the ZIP from ChatGPT and upload it into:

```text
/workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo/
```

Then:

```bash
cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo
unzip -o nfl_app_v4_6_top2_odds.zip
```

## Export real data

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

Then:

```bash
cd mobile
npx expo start --tunnel -c
```
