# Model v0.2

## What changed

### 1. Exponential recency weighting
The baseline no longer treats every prior game equally.

Default half-life: 6 team-games.

That means recent games matter more while older games still contribute.

### 2. Opponent-adjusted EPA
Raw offensive EPA is adjusted by the defense faced.

`adj_off_epa = game_off_epa - opponent_pregame_def_epa`

Raw defensive EPA allowed is adjusted by opponent offensive strength.

`adj_def_epa = game_def_epa - opponent_pregame_off_epa`

The same approach is applied to success rate.

### 3. No-leakage construction
Opponent strength is always taken from the opponent's PRE-game rating.
The current game's performance never appears inside its own prediction features.

## Why this is better

A +0.15 EPA/play offensive performance against an elite defense should count
more than the same +0.15 against a weak defense.

Likewise, recent form should generally matter more than equally weighted games
from months earlier.

## Next upgrade: v0.3 QB model

Recommended next:
- QB EPA/dropback
- CPOE
- sack avoidance
- explosive pass rate
- starter identification
- replacement-level QB baseline
- QB change adjustment
- rolling / decayed QB form
