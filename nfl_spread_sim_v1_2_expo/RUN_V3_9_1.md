# v3.9.1 Closing-Capture Guard

This is a safety patch for v3.9.

The original `close` command could update every qualifier in the selected
season/week. NFL games can have different kickoff times, so one near-kickoff
pull must not be labeled as the close for later games.

v3.9.1 makes `--game-id` mandatory.

## Install

```bash
unzip -o nfl_spread_totals_v3_9_1_close_guard.zip
source .venv-model/bin/activate
python -m pytest v39_tests v391_tests -q
```

## Capture close candidate for ONE game

First refresh odds into a separate file:

```bash
python -m score_model.multi_book_live_v31 \
  --season 2026 \
  --week 3 \
  --max-age-minutes 15 \
  --all-candidates-output artifacts/live/week3_2026_closing_candidates_v39.csv \
  --best-output artifacts/live/week3_2026_closing_best_v39.csv
```

Then update only the game actually near kickoff:

```bash
python -m score_model.forward_test_v39 close \
  --all-candidates-csv artifacts/live/week3_2026_closing_candidates_v39.csv \
  --game-id 2026_03_KC_MIA \
  --season 2026 \
  --week 3 \
  --snapshot-label closing_candidate
```

For games sharing the same kickoff window, repeat `--game-id`:

```bash
python -m score_model.forward_test_v39 close \
  --all-candidates-csv artifacts/live/week3_2026_closing_candidates_v39.csv \
  --game-id 2026_03_KC_MIA \
  --game-id 2026_03_LAC_BUF \
  --season 2026 \
  --week 3 \
  --snapshot-label closing_candidate
```

Only include games that are actually near their own kickoff.

## Current ledger

The existing ledger created by v3.9 is preserved. Installing this patch does
not modify or duplicate any previously captured qualifier.
