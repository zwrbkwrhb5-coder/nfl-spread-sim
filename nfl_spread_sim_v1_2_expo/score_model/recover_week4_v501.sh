#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo

if [ -f .venv-model/bin/activate ]; then
  . .venv-model/bin/activate
fi

BAD_BEST="artifacts/live/week3_2026_best_v31.csv"
BAD_ALL="artifacts/live/week3_2026_all_candidates_v31.csv"
W4_BEST="artifacts/live/week4_2026_best_v31.csv"
W4_ALL="artifacts/live/week4_2026_all_candidates_v31.csv"

if [ ! -f "$BAD_BEST" ]; then
  echo "Missing $BAD_BEST"
  exit 1
fi
if [ ! -f "$BAD_ALL" ]; then
  echo "Missing $BAD_ALL"
  exit 1
fi

# Confirm the mislabeled files really contain Week 4 before copying them.
python - <<'PY'
import pandas as pd
from pathlib import Path
for p in [Path('artifacts/live/week3_2026_best_v31.csv'), Path('artifacts/live/week3_2026_all_candidates_v31.csv')]:
    df = pd.read_csv(p)
    if 'game_id' not in df.columns or df.empty:
        raise SystemExit(f'{p} does not contain usable game_id rows')
    ids = df['game_id'].astype(str)
    if not ids.str.startswith('2026_04_').all():
        bad = ids[~ids.str.startswith('2026_04_')].head().tolist()
        raise SystemExit(f'{p} is not the mislabeled Week 4 output. Example unexpected ids: {bad}')
    print(f'Confirmed Week 4 data in mislabeled file: {p} ({len(df)} rows)')
PY

cp "$BAD_BEST" "$W4_BEST"
cp "$BAD_ALL" "$W4_ALL"

echo "Recovered Week 4 CSV filenames."
echo "Combining preserved Week 3 app snapshot with Week 4 predictions..."
python -m score_model.export_mobile_v501 \
  --season 2026 \
  --current-week 3 \
  --next-week 4 \
  --current-snapshot mobile/generatedData.ts \
  --next-best "$W4_BEST" \
  --next-all-candidates "$W4_ALL" \
  --output mobile/generatedData.ts

echo "Done. Open the app and select WEEK 4 / NEXT."
