# Market / Odds Intelligence Layer v0.9

## Purpose

Turn model probabilities into sportsbook-specific spread analysis.

## Data tracked

For every sportsbook and snapshot:
- game
- timestamp
- sportsbook
- side
- spread
- American odds
- market phase

Recommended phases:
- opening
- current
- closing

## Derived values

### Break-even probability

Example:
-110 => 52.38%

### Consensus spread

Median spread across books.

### Line movement

Track:
- opening -> current
- opening -> close

### Projected differential

From the selected team's perspective:

`model margin + sportsbook spread`

Example:
Model BUF +6.2
Book BUF -3.5

Projected differential = +2.7 points

### Probability edge

`calibrated simulation cover probability - sportsbook break-even probability`

Example:
Calibrated cover = 58.9%
Break-even = 51.22%

Probability edge = +7.68 percentage points

## Best book selection

For the same side:
1. Prefer the most favorable spread.
2. If spread is equal, prefer the better price.

Examples:
- favorite -2.5 is better than -3.5
- underdog +4.5 is better than +3.5

## Top 5 ranking

Primary ranking:
- calibrated probability edge

Secondary display:
- projected differential

This prevents the app from ranking opportunities only by raw spread difference.

## Closing-line value

Once historical closing data exists, compare the line available when the
model generated a pick against the closing line.

That gives us a separate quality measure:
- did our preferred side beat the closing market?
- how often?
- by how many points?

## Anti-leakage

Closing line may only be used for retrospective evaluation.

It must never be fed into a prediction made before the close unless the
prediction timestamp is after that line existed.
