#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo

if [ -f .venv-model/bin/activate ]; then
  . .venv-model/bin/activate
fi

echo "[1/2] Running the existing v3.1 model for Week 4..."
python -m score_model.multi_book_live_v31 \
  --season 2026 \
  --week 4 \
  --max-age-minutes 30

echo "[2/2] Combining Week 3 + Week 4 for the app..."
python -m score_model.export_mobile_v50 \
  --season 2026 \
  --current-week 3 \
  --next-week 4 \
  --current-best artifacts/live/week3_2026_best_v31.csv \
  --current-all-candidates artifacts/live/week3_2026_all_candidates_v31.csv \
  --next-best artifacts/live/week4_2026_best_v31.csv \
  --next-all-candidates artifacts/live/week4_2026_all_candidates_v31.csv \
  --ledger artifacts/live/forward_test_ledger_v39.csv \
  --results artifacts/live/week3_2026_results.csv \
  --historical artifacts/market_stack_v37/overall_metrics.csv \
  --output mobile/generatedData.ts

echo "Done. Week 3 + Week 4 are now in mobile/generatedData.ts"
