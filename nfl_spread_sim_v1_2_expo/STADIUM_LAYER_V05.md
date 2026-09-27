# Stadium / Environment Layer v0.5

v0.5 tests whether stadium environment improves the team + QB model.

## Initial features

- loudness index
- loudness per 1,000 seats
- capacity
- indoor / outdoor
- altitude

## Interactions

- loudness × visiting QB sack rate (when available)
- loudness × visiting QB CPOE (when available)
- loudness × indoor stadium

## Important design rule

No stadium receives a manually assigned point bonus.

The model must demonstrate out-of-sample improvement.

## Starter loudness data

The included CSV begins with the stadium rankings previously supplied by the
user. Missing stadiums/loudness values should be filled with verified data
before a full-league backtest.

## Fair test

v0.5 compares:
1. team + QB
2. team + QB + stadium/environment

on the exact same set of games.

Negative delta MAE/RMSE means the environment layer improved error.

## Future environment additions

- actual attendance / capacity %
- wind
- temperature
- precipitation
- roof status
- playing surface
- travel distance
- time-zone/body-clock shift
- rest days / short week / bye
- away false-start/pre-snap penalty history
