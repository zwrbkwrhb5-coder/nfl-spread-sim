# v2.3 Live First Bet Runner

This runs the current QB-enhanced spread/totals model across the 2026 Week 3 slate.

Important:
- The included market CSV is only an initial snapshot.
- Update lines/prices immediately before running if your sportsbook differs.
- A ranked model edge is not proof of profitability.
- This is forward testing of an unproven live model.

## 1. Install

```bash
unzip -n nfl_spread_totals_v2_3_live_first_bet.zip
source .venv-model/bin/activate
python -m pytest live_tests/test_live_v23.py -q
```

## 2. Check/update market snapshot

```bash
cat live_data/week3_2026_market.csv
```

Column convention:
- `home_spread=-7` means home team -7
- `home_spread=+3` means home team +3

Edit any line or price to match the sportsbook you actually intend to use.

## 3. Run today's slate

```bash
python -m score_model.live_first_bet_v23 \
  --training-csv artifacts/dual_market/training_games_qb_v17.csv \
  --oos-csv artifacts/dual_market/oos_qb_v18_common.csv \
  --market-csv live_data/week3_2026_market.csv \
  --season 2026 \
  --week 3 \
  --output artifacts/live/week3_2026_ranked.csv
```

Paste the TOP 10 output back into ChatGPT before placing anything.

The runner:
- loads completed 2025 and 2026 play-by-play
- builds current pregame team form
- identifies the most recent primary QB
- applies 100-dropback QB shrinkage
- predicts margin and total
- estimates probabilities from historical OOS residuals
- compares each option to sportsbook break-even
- ranks the slate by model edge
