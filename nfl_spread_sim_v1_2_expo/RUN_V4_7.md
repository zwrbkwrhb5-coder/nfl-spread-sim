# NFL SIM v4.7 — Best List

This version adds a persistent personal Best List.

## New behavior

Every #1 / #2 sportsbook option now has a button:

```text
+ ADD TO BEST LIST
```

After saving:

```text
✓ SAVED TO BEST LIST
```

The exact:
- matchup
- market
- pick
- line
- odds
- sportsbook
- model edge
- model probability
- break-even probability

is saved.

The Picks screen now has two tabs:

```text
MODEL PICKS | MY BEST LIST
```

Saved bets survive app restarts using AsyncStorage.

## Install

Download the ZIP from ChatGPT and upload it into:

```text
/workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo/
```

Then:

```bash
cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo
unzip -o nfl_app_v4_7_best_list.zip

cd mobile
npx expo install @react-native-async-storage/async-storage
```

You do NOT need to regenerate model data solely for this UI update if v4.6 data is
already working.

Then start Expo:

```bash
npx expo start --tunnel -c
```
