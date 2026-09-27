#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f .venv-model/bin/activate ]; then
  . .venv-model/bin/activate
fi

STATUS_ONLY=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --status) STATUS_ONLY=1 ;;
    --force) FORCE=1 ;;
    *) echo "Unknown option: $arg"; exit 2 ;;
  esac
done

eval "$(python -m score_model.nfl_week_v51 --shell)"

if [ "$SEASON" -ne 2026 ]; then
  echo "This project is currently configured for the 2026 NFL season only. Detected season: $SEASON"
  exit 1
fi

SNAPSHOT="mobile/generatedData.ts"

read_snapshot_status() {
  python - "$SNAPSHOT" "$CURRENT_WEEK" "$NEXT_WEEK" <<'PY_STATUS'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
current = int(sys.argv[2])
next_week = int(sys.argv[3])
if not p.exists():
    print("SNAPSHOT_CURRENT=0")
    print("HAS_CURRENT=0")
    print("HAS_NEXT=0")
    raise SystemExit
text = p.read_text(encoding='utf-8')
marker = 'export const APP_DATA = '
if marker not in text:
    print("SNAPSHOT_CURRENT=0")
    print("HAS_CURRENT=0")
    print("HAS_NEXT=0")
    raise SystemExit
payload = text.split(marker,1)[1]
payload = payload.rsplit(' satisfies AppSnapshot;',1)[0].strip().rstrip(';')
data = json.loads(payload)
games = data.get('games', [])
print(f"SNAPSHOT_CURRENT={int(data.get('currentWeek') or data.get('week') or 0)}")
print(f"HAS_CURRENT={1 if any(int(g.get('week',-1)) == current for g in games) else 0}")
print(f"HAS_NEXT={1 if next_week and any(int(g.get('week',-1)) == next_week for g in games) else 0}")
PY_STATUS
}

eval "$(read_snapshot_status)"

echo "NFL SIM AUTO WEEK STATUS"
echo "Season: $SEASON"
echo "Detected current week: $CURRENT_WEEK"
if [ "$NEXT_WEEK" -gt 0 ]; then
  echo "Detected next week: $NEXT_WEEK"
else
  echo "Detected next week: none (Week 18)"
fi
echo "App snapshot current week: $SNAPSHOT_CURRENT"
echo "Current week loaded: $HAS_CURRENT"
echo "Next week loaded: $HAS_NEXT"

if [ "$STATUS_ONLY" -eq 1 ]; then
  exit 0
fi

if [ "$FORCE" -eq 0 ] && [ "$SNAPSHOT_CURRENT" -eq "$CURRENT_WEEK" ] && [ "$HAS_CURRENT" -eq 1 ]; then
  if [ "$NEXT_WEEK" -eq 0 ] || [ "$HAS_NEXT" -eq 1 ]; then
    echo "No rollover needed. CURRENT + NEXT are already loaded. No Odds API requests used."
    exit 0
  fi
fi

if [ -z "${THE_ODDS_API_KEY:-}" ]; then
  echo "THE_ODDS_API_KEY is not set."
  echo "For Codespaces: export THE_ODDS_API_KEY=..."
  echo "For GitHub Actions: add repository secret THE_ODDS_API_KEY."
  exit 1
fi

run_week() {
  local week="$1"
  echo "Running v3.1 model for Week $week..."
  python -m score_model.multi_book_live_v31 \
    --season "$SEASON" \
    --week "$week" \
    --max-age-minutes 30

  python - "$SEASON" "$week" <<'PY_COPY'
import shutil, sys
from pathlib import Path
import pandas as pd
season = int(sys.argv[1])
week = int(sys.argv[2])
prefix = f"{season}_{week:02d}_"
pairs = [
    (Path('artifacts/live/week3_2026_best_v31.csv'), Path(f'artifacts/live/week{week}_{season}_best_v31.csv')),
    (Path('artifacts/live/week3_2026_all_candidates_v31.csv'), Path(f'artifacts/live/week{week}_{season}_all_candidates_v31.csv')),
]
for src, dst in pairs:
    if not src.exists():
        raise SystemExit(f'Model output missing: {src}')
    df = pd.read_csv(src)
    if df.empty or 'game_id' not in df.columns:
        raise SystemExit(f'Model output is empty or invalid: {src}')
    ids = df['game_id'].astype(str)
    if not ids.str.startswith(prefix).all():
        bad = ids[~ids.str.startswith(prefix)].head().tolist()
        raise SystemExit(f'Expected Week {week} rows in {src}; unexpected ids: {bad}')
    if src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f'Saved Week {week}: {dst} ({len(df)} rows)')
PY_COPY
}

CURRENT_BEST="artifacts/live/week${CURRENT_WEEK}_${SEASON}_best_v31.csv"
CURRENT_ALL="artifacts/live/week${CURRENT_WEEK}_${SEASON}_all_candidates_v31.csv"
NEXT_BEST=""
NEXT_ALL=""

if [ "$FORCE" -eq 1 ] || [ "$SNAPSHOT_CURRENT" -ne "$CURRENT_WEEK" ] || [ "$HAS_CURRENT" -ne 1 ]; then
  run_week "$CURRENT_WEEK"
else
  CURRENT_BEST=""
  CURRENT_ALL=""
fi

NEXT_OK=0
if [ "$NEXT_WEEK" -gt 0 ]; then
  NEXT_BEST="artifacts/live/week${NEXT_WEEK}_${SEASON}_best_v31.csv"
  NEXT_ALL="artifacts/live/week${NEXT_WEEK}_${SEASON}_all_candidates_v31.csv"
  if [ "$FORCE" -eq 1 ] || [ "$SNAPSHOT_CURRENT" -ne "$CURRENT_WEEK" ] || [ "$HAS_NEXT" -ne 1 ]; then
    if run_week "$NEXT_WEEK"; then
      NEXT_OK=1
    else
      echo "Week $NEXT_WEEK is not available yet. The app will keep Week $CURRENT_WEEK and retry on the next scheduled check."
      NEXT_BEST=""
      NEXT_ALL=""
    fi
  else
    NEXT_OK=1
  fi
fi

ARGS=(
  --season "$SEASON"
  --current-week "$CURRENT_WEEK"
  --next-week "$NEXT_WEEK"
  --current-snapshot "$SNAPSHOT"
  --output "$SNAPSHOT"
)
if [ -n "$CURRENT_BEST" ] && [ -f "$CURRENT_BEST" ]; then
  ARGS+=(--current-best "$CURRENT_BEST" --current-all-candidates "$CURRENT_ALL")
fi
if [ -n "$NEXT_BEST" ] && [ -f "$NEXT_BEST" ]; then
  ARGS+=(--next-best "$NEXT_BEST" --next-all-candidates "$NEXT_ALL")
fi

echo "Rebuilding mobile CURRENT + NEXT board..."
python -m score_model.export_mobile_v51 "${ARGS[@]}"

echo "AUTO ROLLOVER COMPLETE"
echo "Current: Week $CURRENT_WEEK"
if [ "$NEXT_WEEK" -gt 0 ] && [ "$NEXT_OK" -eq 1 ]; then
  echo "Next: Week $NEXT_WEEK"
fi
