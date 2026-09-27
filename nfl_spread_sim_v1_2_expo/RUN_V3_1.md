# v3.1 — Indiana Multi-Book Live Shopping

v3.1 keeps the v2.9 model and v3.0 input gates, then shops every Indiana
sportsbook currently exposed by The Odds API.

## Indiana mobile books tracked

The code recognizes:

- Bally Bet
- bet365
- BetMGM
- BetRivers
- Caesars
- DraftKings
- Fanatics
- FanDuel
- Hard Rock Bet
- SBK
- theScore Bet

Current The Odds API mappings in v3.1:

- Bally Bet -> `ballybet`
- BetMGM -> `betmgm`
- BetRivers -> `betrivers`
- Caesars -> `williamhill_us`
- DraftKings -> `draftkings`
- Fanatics -> `fanatics`
- FanDuel -> `fanduel`
- Hard Rock Bet -> `hardrockbet`
- theScore Bet -> `espnbet`

The Odds API's published US/US2 list does not currently list Indiana bet365
or SBK. v3.1 reports those as coverage gaps instead of pretending their prices
are present.

## Security

Do NOT paste your API key into ChatGPT, source files, or Git commits.

For the current terminal session:

```bash
export THE_ODDS_API_KEY='PASTE_YOUR_KEY_HERE'
```

A Codespaces secret is better for persistence.

## Install

```bash
unzip -n nfl_spread_totals_v3_1_indiana_multibook.zip
source .venv-model/bin/activate
python -m pytest v31_tests -q
```

## Run

```bash
python -m score_model.multi_book_live_v31 \
  --season 2026 \
  --week 3 \
  --max-age-minutes 30
```

## What it does

1. Calls The Odds API for NFL spreads and totals across `us,us2`.
2. Keeps only sportsbook brands authorized for the Indiana mobile market.
3. Archives every raw spread/total quote with its source and update time.
4. Matches quotes to the nflverse Week 3 schedule.
5. Runs the v3.0 starter/weather/roof readiness gates.
6. Produces one model margin and total per game using the v2.9 raw-QB core.
7. Scores every book's actual line AND juice separately.
8. Applies the historical isotonic calibration and probability shrinkage.
9. Chooses the highest calibrated edge per game/market.
10. Refuses to qualify stale quotes or inputs that fail readiness.

## Outputs

All sportsbook candidates:

```text
artifacts/live/week3_2026_all_candidates_v31.csv
```

Best price/edge per game and market:

```text
artifacts/live/week3_2026_best_v31.csv
```

Raw quote snapshots for later CLV work:

```text
artifacts/live/odds_snapshots/odds_v31_<UTC timestamp>.csv
```

## Important

"Best available line" is NOT chosen just by the point spread.

For every candidate, v3.1 recomputes:

- raw cover/over probability at that exact line
- calibrated probability
- final shrunk probability
- break-even probability from that exact American price
- final edge

That means, for example, `+3 -105` can correctly beat `+3.5 -125` if the
model's probability/price tradeoff says it has the higher edge.

The model threshold is not loosened to force action. `QUALIFYING BETS: NONE`
is a valid live result.
