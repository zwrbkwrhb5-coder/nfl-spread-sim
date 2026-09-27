# NFL Spread Simulator v1.0 — End-to-End Orchestration

## What v1.0 does

Given:
- one matchup feature vector
- current sportsbook spread/price data
- a trained margin model
- a trained uncertainty model

the pipeline returns:

1. projected home margin
2. projected winner
3. matchup-specific volatility
4. sportsbook-by-sportsbook spread analysis
5. simulated cover probability
6. implied break-even probability
7. probability edge
8. projected differential
9. best available sportsbook opportunity
10. full 1,000,000-simulation rerun for the best line

Across a slate, it also builds:
- Top 5 spread opportunities

## Important limitation

v1.0 is the first orchestration layer, not yet a fully automatic live-data app.

Several upstream sources are still templates/experimental:
- injuries
- stadium completeness
- weather
- current sportsbook feed
- college player priors

The pipeline is ready to consume those sources once populated.

## Game command

```bash
python -m src.run_game \
  --features data/example_matchup_features.csv \
  --market data/market_odds_template.csv \
  --game-id example_game \
  --home BUF \
  --away MIA
```

## Slate command

```bash
python -m src.run_slate \
  --manifest data/slate_manifest_template.csv \
  --market data/market_odds_template.csv
```

## User-facing result fields

Recommended app display:

- Model projection
- Market spread
- Projected differential
- Cover probability
- Break-even probability
- Probability edge
- Best sportsbook
- 1,000,000 simulations
- Margin range
