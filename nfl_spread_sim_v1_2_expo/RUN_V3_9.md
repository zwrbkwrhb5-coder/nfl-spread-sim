# v3.9 Forward-Test + CLV Ledger

v3.8 confirmed the historical market file is a closing-line dataset.

That means the cleanest next step is not more retrospective threshold tuning.
It is prospective logging.

v3.9 does NOT change the v3.1 model.

It records exactly what v3.1 qualified at the time of the pull, distinguishes
"model qualified" from "actually placed", captures a later closing snapshot,
and calculates line/price CLV.

## Current status from the v3.8 run

The fresh v3.1 pull produced three qualifiers:

```text
KC @ MIA  UNDER 50.5 -110  DraftKings
KC @ MIA  MIA +13.5 +100   FanDuel
LAC @ BUF UNDER 52.5 -110  theScore Bet
```

These are model qualifiers, NOT automatically marked as wagers.

The v3.8 totals stack remains shadow-only because it was trained against
historical closing totals while the live input is a current multi-book
consensus.

## Install

```bash
unzip -n nfl_spread_totals_v3_9_forward_test_clv.zip
source .venv-model/bin/activate
python -m pytest v39_tests -q
```

## 1. Capture the current v3.1 qualifiers

```bash
python -m score_model.forward_test_v39 capture \
  --best-csv artifacts/live/week3_2026_best_v31.csv \
  --season 2026 \
  --week 3 \
  --snapshot-label current
```

This appends to:

```text
artifacts/live/forward_test_ledger_v39.csv
```

It does NOT assume a qualifier was actually bet.

## 2. Optional: mark an actually placed wager

Example only:

```bash
python -m score_model.forward_test_v39 mark \
  --game-id 2026_03_KC_MIA \
  --market total \
  --pick UNDER \
  --stake-units 1
```

Only run `mark` for wagers you actually placed.

## 3. Capture a near-close market snapshot later

First run v3.1 again to a separate file:

```bash
python -m score_model.multi_book_live_v31 \
  --season 2026 \
  --week 3 \
  --max-age-minutes 15 \
  --all-candidates-output artifacts/live/week3_2026_closing_candidates_v39.csv \
  --best-output artifacts/live/week3_2026_closing_best_v39.csv
```

Then compare the original qualifiers against that snapshot:

```bash
python -m score_model.forward_test_v39 close \
  --all-candidates-csv artifacts/live/week3_2026_closing_candidates_v39.csv \
  --season 2026 \
  --week 3 \
  --snapshot-label closing_candidate
```

Do not label it a true close unless the pull was actually taken at market
close / immediately before kickoff.

## CLV fields

For spreads:

```text
positive clv_points = original selected-team handicap was better
```

Examples:

```text
MIA +13.5 entry vs +12.5 close = +1.0 point CLV
PHI -2.5 entry vs -3.0 close   = +0.5 point CLV
```

For totals:

```text
OVER: lower entry total is better
UNDER: higher entry total is better
```

Same-line price CLV is also recorded when the closing snapshot still has the
exact entry line.

## 4. Report the ledger

```bash
python -m score_model.forward_test_v39 report \
  --season 2026 \
  --week 3
```

## 5. Settle later

When a results CSV exists with:

```text
game_id,actual_margin,actual_total
```

run:

```bash
python -m score_model.forward_test_v39 settle \
  --results-csv PATH_TO_RESULTS.csv \
  --season 2026 \
  --week 3
```

## Research rules

- v3.1 remains the production/live model.
- v3.8 totals stack remains shadow-only.
- The 5+ total-gap rule remains rejected.
- Do not change the 1% edge threshold during this forward-test sample.
- Do not retroactively change entry lines or books.
- Keep "qualified" separate from "actually placed".
- Evaluate CLV and results after enough prospective observations accumulate.
