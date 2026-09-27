#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo

if [ -f .venv-model/bin/activate ]; then
  . .venv-model/bin/activate
fi

if [ -z "${THE_ODDS_API_KEY:-}" ]; then
  echo "THE_ODDS_API_KEY is not set. Export it in this terminal or add it as a Codespaces secret."
  exit 1
fi

SNAPSHOT_BACKUP="/tmp/mobile_generatedData_before_week4.ts"
cp mobile/generatedData.ts "$SNAPSHOT_BACKUP"

echo "[1/3] Running v3.1 model for Week 4..."
python -m score_model.multi_book_live_v31 \
  --season 2026 \
  --week 4 \
  --max-age-minutes 30

echo "[2/3] Renaming the v3.1 runner's fixed Week-3 output names to Week 4..."
python - <<'PY'
import pandas as pd
from pathlib import Path
pairs = [
    (Path('artifacts/live/week3_2026_best_v31.csv'), Path('artifacts/live/week4_2026_best_v31.csv')),
    (Path('artifacts/live/week3_2026_all_candidates_v31.csv'), Path('artifacts/live/week4_2026_all_candidates_v31.csv')),
]
for src, dst in pairs:
    df = pd.read_csv(src)
    if 'game_id' not in df.columns or df.empty or not df['game_id'].astype(str).str.startswith('2026_04_').all():
        raise SystemExit(f'Expected Week 4 rows in {src}, but validation failed.')
    dst.write_bytes(src.read_bytes())
    print(f'Wrote {dst} ({len(df)} rows)')
PY

echo "[3/3] Combining preserved Week 3 snapshot with fresh Week 4 predictions..."
python -m score_model.export_mobile_v501 \
  --season 2026 \
  --current-week 3 \
  --next-week 4 \
  --current-snapshot "$SNAPSHOT_BACKUP" \
  --next-best artifacts/live/week4_2026_best_v31.csv \
  --next-all-candidates artifacts/live/week4_2026_all_candidates_v31.csv \
  --output mobile/generatedData.ts

echo "Done. Week 3 + Week 4 are in mobile/generatedData.ts"
