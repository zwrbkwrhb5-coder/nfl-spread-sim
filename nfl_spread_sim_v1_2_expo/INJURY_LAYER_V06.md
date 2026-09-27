# Player Availability / Injury Layer v0.6

## Goal

Measure the effect of WHO is unavailable, not simply how many players are hurt.

## Player-level inputs

- position
- injury / availability status
- prior snap share
- starter flag
- player value prior

## Status severity

Starting defaults:
- out / IR / inactive = 1.00
- doubtful = 0.75
- questionable = 0.35
- limited = 0.20
- probable = 0.10
- active / full = 0.00

These are not point-spread adjustments.
They only create model inputs.

## Position importance starting priors

- QB = 1.00
- OL = 0.40
- WR/TE = 0.35
- pass rush = 0.35
- secondary = 0.30
- interior DL = 0.20
- LB = 0.18
- RB = 0.12
- special teams = 0.08

These are intentionally starting priors and should be tuned or replaced
through historical backtesting.

## Team-game features

Examples:
- total injury impact
- out/inactive count
- questionable/limited count
- QB injury impact
- OL injury impact
- WR/TE injury impact
- pass-rush injury impact
- secondary injury impact

The game model receives HOME minus AWAY injury burden.

## Why this is better than injury count

Losing:
- QB1
- LT1
- WR1

should not be treated the same as losing:
- RB3
- special teams reserve
- backup linebacker

## Important anti-leakage rule

Every player-value estimate and snap-share estimate must come from information
available before kickoff.

No current-game snaps or performance may influence that game's injury feature.

## Next upgrades

- official inactive status
- practice participation
- expected snap reduction
- OL continuity
- WR target-share concentration
- pass-rush pressure share
- secondary coverage role
- learned WAR-like player values
- interaction with QB injury-return layer
