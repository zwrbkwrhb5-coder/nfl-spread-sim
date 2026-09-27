# NFL Spread Simulator v1.0

This is the first end-to-end orchestration release.

## Pipeline

Pregame features
→ projected margin
→ matchup-specific uncertainty
→ Monte Carlo simulations
→ sportsbook comparison
→ calibrated probability edge
→ best line
→ Top 5 slate opportunities

## Single-game output

The pipeline is designed to return:

- projected winner
- projected scoring margin
- matchup volatility
- best sportsbook
- best spread
- American odds
- projected differential
- cover probability
- break-even probability
- probability edge
- 1,000,000 simulation rerun

## Slate output

`artifacts/top5_v1_0.csv`

## Important

v1.0 connects the research modules into one workflow.

It still depends on trained artifacts and populated real-world data feeds.
The remaining engineering work is live-data automation and app UI integration.
