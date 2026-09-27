# QB Layer v0.3

## New components

### QB form
Per-game QB metrics:
- EPA per dropback
- success rate
- CPOE
- sack rate
- explosive pass rate
- interception rate

The QB form model uses exponentially weighted recent performance.

### College-to-NFL prior
College data can be blended into inexperienced NFL QBs.

College weight fades as NFL dropbacks accumulate.

The starting blend curve is intentionally simple and should be tuned with
historical backtests.

### First-game-back injury layer
This is a separate QB adjustment.

For each historical injury return, we record:
- injury type
- games missed
- QB's pre-injury baseline
- first-game-back EPA/dropback
- first-game-back success rate
- first-game-back CPOE
- first-game-back sack rate
- change vs pre-injury baseline

Historical events are grouped by:
- injury type
- duration bucket: 1, 2-3, 4-6, 7+ games missed

The adjustment is shrunk toward zero when sample sizes are small.

## Critical anti-leakage rule

When predicting a QB's first game back, the outcome of that return game is NOT
available to the model.

Only older completed injury-return cases may contribute to the prior.

## Future improvements

The injury model should eventually distinguish:
- throwing shoulder / elbow / hand
- non-throwing upper body
- ankle / foot
- knee
- hamstring / groin
- ribs / torso
- concussion
- illness / conditioning

It should also account for:
- practice participation before return
- whether QB was limited or full
- age
- scrambling dependence
- offensive-line quality
- opponent pass rush
- weather
- whether injury affects throwing or mobility
