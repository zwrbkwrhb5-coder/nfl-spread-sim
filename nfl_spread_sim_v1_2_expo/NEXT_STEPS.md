# Next model upgrades

## v0.2 — opponent-adjusted team strength
- Replace raw rolling EPA with opponent-adjusted EPA.
- Add exponentially decayed recency weights.
- Add preseason priors / prior-season carryover.
- Separate early-down passing from garbage-time production.

## v0.3 — quarterback model
- Build QB rolling EPA / CPOE / sack avoidance / explosive-pass contribution.
- Estimate replacement-level QB.
- Add starter-change adjustment.

## v0.4 — context
- Rest days
- short week
- bye week
- travel/time zone
- dome/outdoor
- wind/weather

## v0.5 — market layer
- Store opening, current and closing spreads.
- Train a raw-football model and a market-aware model separately.
- Measure closing-line value and model-vs-market residuals.

## v0.6 — distribution calibration
- Stop assuming one residual distribution fits every matchup.
- Learn heteroskedastic uncertainty.
- Calibrate cover probability by spread/key-number bucket.
- Compare bootstrap residual simulation with quantile models.

## v0.7 — player availability
- Injury / inactive feed.
- Offensive line continuity.
- WR/TE target share availability.
- pass rush / secondary availability.

## Core scorecard
Every new feature must improve walk-forward results, not in-sample fit.

Track:
- MAE of game margin
- RMSE
- ATS accuracy vs closing spread
- Brier score for cover probabilities
- calibration curve
- log loss
- closing-line value
- performance by confidence bucket
