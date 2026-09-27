# Game Context Layer v0.7

## New features

### Schedule / rest
- home rest days
- away rest days
- rest-day differential
- short-week flags
- off-bye flags

### Travel
- team-to-stadium travel miles
- home vs away travel differential
- time-zone shift
- travel × time-zone interaction

### Game type
- divisional game flag

### Surface / roof
- grass
- turf/artificial
- indoor/dome
- roof closed

### Weather
- temperature
- wind
- precipitation

### Interactions
- wind × outdoor
- travel × time-zone shift
- wind × visiting-QB CPOE when QB feature is available
- short week × visiting-QB sack rate when available

## Design rules

No factor receives a manual point adjustment.

Rest, wind, travel, turf, division games, etc. are merely model inputs.
They remain only if they improve honest walk-forward performance or calibration.

## Anti-leakage

Only information known before kickoff may be used:
- scheduled date
- travel location
- historical rest
- forecast / known roof status at prediction time
- current surface
- divisional relationship

For weather backtests, use a historical pregame forecast snapshot if possible.
Actual final-game weather should not be substituted if the model would not have
known it before kickoff.

## A/B test

v0.6:
Team + QB + stadium + injury

vs.

v0.7:
v0.6 + game context

Negative delta MAE/RMSE indicates improvement on the exact same test games.
