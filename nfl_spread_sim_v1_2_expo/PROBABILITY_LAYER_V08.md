# Probability / Uncertainty Layer v0.8

## Problem with earlier simulation

Earlier versions use one global historical residual distribution.

That means:
- elite QB vs elite QB
- rookie QB in bad weather
- heavily injured team
- divisional low-variance matchup
- explosive mismatch

all receive roughly the same baseline volatility.

That is too crude.

## v0.8 solution

### 1. Learn game-specific uncertainty

Train a second model whose target is:

`absolute margin error`

using only honest out-of-sample prediction residuals.

The uncertainty model sees the same pregame feature set and learns which
matchups historically produce larger or smaller errors.

### 2. Convert expected absolute error to sigma

For a normal approximation:

`E|X| = sigma * sqrt(2/pi)`

That gives us a matchup-specific volatility estimate.

### 3. Preserve empirical tails

We do NOT simply switch to a normal distribution.

Historical residuals are rescaled to the game-specific sigma and bootstrapped.

This preserves:
- asymmetry
- fat tails
- blowout behavior
better than a pure Gaussian assumption.

### 4. Probability calibration

Raw Monte Carlo cover probability can still be overconfident or underconfident.

v0.8 includes isotonic calibration support so historical raw cover estimates
can be mapped to actual observed cover frequency.

Example:

raw model says 60%
historically those cases cover 57.2%

calibrated probability = 57.2%

## Desired validation

We ultimately want:
- Brier score
- log loss
- calibration curve
- reliability by probability bucket
- sharpness
- ATS results by calibrated edge
- coverage of 50/80/90/95% margin intervals

## Core rule

A model that predicts the average margin well but is badly calibrated is not
good enough for spread analysis.

Probability quality matters as much as point prediction.
