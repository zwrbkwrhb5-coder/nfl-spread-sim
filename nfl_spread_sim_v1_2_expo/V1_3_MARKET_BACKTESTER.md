# v1.3 Historical Market Backtester — Spread + Totals

This add-on evaluates the model against historical sportsbook lines.

## What it adds

- ATS grading for spread bets
- Over/Under grading
- push handling
- American-odds break-even probabilities
- unit profit / ROI
- empirical probabilities from out-of-sample residual distributions
- walk-forward isotonic probability calibration
- edge buckets
- closing-line-value diagnostics
- Ridge alpha tuning helper

## Required inputs

### 1. OOS model predictions

`data/oos_predictions_template.csv`

Required fields:

- game_id
- season
- pred_home_score
- pred_away_score
- pred_margin
- pred_total
- actual_home_score
- actual_away_score

### 2. Historical sportsbook market lines

`data/historical_market_template.csv`

Required:

- game_id
- season
- home_team
- away_team
- home_spread
- home_spread_odds
- total_line
- over_odds
- under_odds
- home_score
- away_score

Optional closing-line fields:

- opening_home_spread
- closing_home_spread
- closing_total

## Run

```bash
python -m score_model.historical_market \
  --market-csv data/historical_market.csv \
  --oos-predictions artifacts/dual_market/oos_predictions.csv \
  --output-dir artifacts/market_backtest_v13 \
  --min-edge 0.00
```

Outputs:

- report.json
- all_market_rows.csv
- best_side_per_game.csv
- bettable_rows.csv
- edge_bucket_summary.csv
- closing_line_summary.csv

## Important limitation

This package does NOT include a licensed historical sportsbook-line database.
You need a real historical line source and must align each quote to the prediction
timestamp. Never use a closing line as an input to a prediction that was supposedly
made earlier in the day.

Positive historical ROI is not proof of future profitability. Treat edge thresholds
as hypotheses that must survive later seasons and untouched holdout periods.
