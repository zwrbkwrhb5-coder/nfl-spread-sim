# v1.4 Notes

What changed:

- automatic nflverse market-data download
- 2018–2025 historical market CSV builder
- real home and away spread prices when available
- real Over / Under prices when available
- explicit -110 fallback flags for missing prices
- proper nflverse spread-sign conversion
- spread backtester now evaluates BOTH home and away sides
- totals backtester evaluates BOTH Over and Under
- each game/market selects the side with the larger calibrated probability edge
- edge bucket ROI output

This makes the first real historical ATS + totals backtest possible without
manually building the market file.
