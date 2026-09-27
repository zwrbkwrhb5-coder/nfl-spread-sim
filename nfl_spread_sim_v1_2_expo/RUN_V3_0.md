# v3.0 Live Input Hardening

v3.0 keeps the v2.9 model core and hardens the live inputs before the model is
allowed to rank a game.

## What v3.0 adds

- current nflverse depth-chart QB1 lookup
- manual QB overrides still take priority
- exact source fields for each QB
- schedule roof/surface/rest metadata
- retractable-roof uncertainty gate
- Open-Meteo outdoor forecast temperature/wind
- market timestamps and maximum line age
- optional The Odds API refresh
- automatic exclusion of stale/unverified games
- audit table before model execution

v3.0 does NOT loosen the model's betting thresholds.

---

## 1. Install + test

```bash
unzip -n nfl_spread_totals_v3_0_live_input_hardening.zip
source .venv-model/bin/activate
python -m pytest v30_tests -q
```

---

## 2A. Recommended: automatic sportsbook refresh

If you have The Odds API key:

```bash
export THE_ODDS_API_KEY='YOUR_KEY'
```

Then run, for example:

```bash
python -m score_model.run_live_v30 \
  --market-csv live_data/week3_2026_market.csv \
  --season 2026 \
  --week 3 \
  --refresh-odds \
  --bookmaker draftkings \
  --market-max-age-minutes 30
```

Use the bookmaker key for the sportsbook whose exact price you intend to bet.

---

## 2B. Manual sportsbook lines

If you are entering lines manually, first replace the old CSV values with the
CURRENT sportsbook lines.

Then explicitly timestamp them:

```bash
python -m score_model.confirm_market_current_v30 \
  --market-csv live_data/week3_2026_market.csv \
  --i-confirm-lines-are-current
```

Do NOT run that command on the old static snapshot just to make it pass the
freshness check.

Then:

```bash
python -m score_model.run_live_v30 \
  --market-csv live_data/week3_2026_market.csv \
  --season 2026 \
  --week 3 \
  --market-max-age-minutes 30
```

---

## 3. Audit only

To inspect inputs without running the betting model:

```bash
python -m score_model.live_inputs_v30 \
  --market-csv live_data/week3_2026_market.csv \
  --season 2026 \
  --week 3 \
  --output live_data/week3_2026_market_hardened_v30.csv
```

The audit shows:

- line age
- market source
- expected QB and source
- roof classification
- temp/wind and source
- spread-ready flag
- totals-ready flag
- exclusion reasons

---

## QB policy

Priority order:

1. `home_qb_id/name` or `away_qb_id/name` manually supplied in market CSV
2. latest nflverse depth-chart QB1
3. no verified QB -> v3.0 excludes the game

The old "most recent primary QB" fallback is no longer enough to pass v3.0's
input gate.

This is deliberate: a wrong upcoming starter can invalidate a live projection.

---

## Roof/weather policy

- fixed indoor/dome: temp=70, wind=0 for consistency with historical context
- outdoor: forecast temp/wind fetched from Open-Meteo
- retractable roof with known `open` or `closed` status: handled normally
- retractable roof with unresolved status: total is excluded
- spread can still run because weather is not part of the spread feature set

You can supply `roof_override=open` or `roof_override=closed` in the market CSV.

---

## Market freshness policy

Default maximum line age: 30 minutes.

A missing timestamp is treated as stale.

The model does not convert a stale snapshot into a current betting
recommendation.

---

## Current architecture

### Spread
- team EPA/success
- raw pregame QB form
- calibrated probability
- 40% probability shrinkage
- 1% minimum edge
- 78-week stability window

### Totals
- totals-specific team features
- raw pregame QB form
- opponent adjustment
- context/weather
- calibrated probability/selection

### Forward testing

The earlier Miami +10.5 and Dallas +3.5 observations remain v2.3-era bets.
New v3.0 qualifying bets should be logged separately.
