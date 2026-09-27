# v1.4 — Automatic Historical NFL Market Data

This add-on pulls the public nflverse `games.csv` data and converts it directly
into the historical-market schema used by the spread + totals backtester.

## Install / test

From your existing project root:

```bash
unzip -n nfl_spread_totals_v1_4_market_data.zip
source .venv-model/bin/activate
python -m pytest market_tests/test_v14_market_data.py -q
```

## Build the historical market file

```bash
python -m score_model.fetch_nflverse_market \
  --start-season 2018 \
  --end-season 2025 \
  --output data/historical_market.csv
```

## Run the real spread + total market backtest

```bash
python -m score_model.market_backtest_v14 \
  --market-csv data/historical_market.csv \
  --oos-predictions artifacts/dual_market/oos_predictions.csv \
  --output-dir artifacts/market_backtest_v14 \
  --min-edge 0.00
```

Then view:

```bash
python -m json.tool artifacts/market_backtest_v14/report.json
```

## Important conventions

nflverse `spread_line` is positive when the HOME team is favored.
This importer converts it to conventional sportsbook notation:

- nflverse spread_line = +3.5
- home_spread = -3.5
- away_spread = +3.5

## Odds

When nflverse contains spread / total prices, the real prices are preserved.
If a price is missing, the importer defaults to -110 and adds an
`*_odds_imputed` flag so imputed prices can be identified in analysis.

## Closing lines

The nflverse schedule data documents `spread_line` and `total_line` as closing
market lines. They are valid for retrospective evaluation / CLV comparisons.
They must not be fed into an earlier prediction as though they were known at
that earlier timestamp.
