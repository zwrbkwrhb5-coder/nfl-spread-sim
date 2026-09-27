# Opening Drive Layer v0.10

## What it studies

### Pregame features
For each team:
- opening-drive points
- opening-drive scoring rate
- opening-drive TD rate
- opening-drive FG rate
- opening-drive turnover rate
- opening-drive EPA

All are shifted and recency weighted.

### Defensive opening-drive features
For each defense:
- opening-drive points allowed
- opening-drive score allowed rate
- opening-drive TD allowed rate
- opening-drive FG allowed rate
- opening-drive turnovers forced
- opening-drive EPA allowed

### Direct matchup signal
Example:
home opening-drive score rate
combined with
away defense opening-drive score allowed rate

This creates a pregame estimate of how favorable the opening drive matchup is.

## Separate ATS research

The project also studies:

ACTUAL home opening-drive result
vs.
final home margin
and, when closing spread data is supplied,
ATS cover result.

This research is descriptive only.

## Critical anti-leakage rule

Actual opening-drive result happens after kickoff.

It may be used to study historical relationships, but it may NEVER be fed into
a pregame prediction for that same game.

Only prior opening-drive tendencies are valid pregame features.

## Why this matters

Two separate questions are being answered:

1. Can opening-drive tendencies help predict the game before kickoff?
2. Historically, how are actual opening-drive results associated with ATS outcomes?

Keeping those separate prevents leakage.
