# NFL SIM v5.1 — Automatic Week Rollover

This patch removes hard-coded Week 3 / Week 4 refresh logic.

## What it does

- Detects the current NFL regular-season week from the Eastern calendar.
- Keeps CURRENT and NEXT in the app.
- On a week rollover, refreshes the new CURRENT week and generates the new NEXT week.
- Corrects the old v3.1 runner's fixed `week3_2026_*` output filenames automatically.
- Gets future kickoff times from a live NFL schedule feed rather than hard-coding Week 5/6/etc.
- Accumulates completed-game scores in `gameResults` for settlement of manually logged bets.
- Is quota-aware: when CURRENT + NEXT are already loaded, it exits before calling The Odds API.
- Includes a GitHub Actions workflow that checks daily; normally it does nothing, and it spends API requests only when a rollover or missing NEXT week requires a model run.

## Install

```bash
cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo
unzip -o nfl_app_v5_1_auto_week_rollover.zip
source .venv-model/bin/activate
bash score_model/refresh_auto_v51.sh --status
```

`--status` does not call The Odds API.

## Manual refresh

```bash
bash score_model/refresh_auto_v51.sh
```

If CURRENT + NEXT are already correct, it exits without using Odds API requests.

Use `--force` only when you intentionally want fresh sportsbook prices for both weeks:

```bash
bash score_model/refresh_auto_v51.sh --force
```

## Enable unattended GitHub refresh

Add the API key as the repository Actions secret `THE_ODDS_API_KEY`.
From Codespaces:

```bash
gh secret set THE_ODDS_API_KEY
```

Paste the key only at the hidden prompt.

Then commit and push:

```bash
git add .github/workflows/nfl-auto-refresh.yml \
  score_model/nfl_week_v51.py \
  score_model/export_mobile_v51.py \
  score_model/refresh_auto_v51.sh

git commit -m "Add automatic NFL week rollover"
git push
```

The workflow then checks daily. On the Tuesday rollover, it changes CURRENT to the new week and generates the following week as NEXT.

## Expo development note

The scheduled job updates `mobile/generatedData.ts` in GitHub. A Codespaces/Expo-Go development session still needs the repository change locally before Metro serves it. A production app can later use an Expo/EAS update channel for fully hands-off phone delivery.
