# NFL SIM v4.8 — Track Only Bets You Actually Place

This removes the Best List behavior.

## New rule

Model picks are recommendations only.

They DO NOT count in your personal tracking record.

A bet is added to your personal Results / Performance tracking only when you tap:

```text
I BET THIS • TRACK 1U
```

After tapping:

```text
✓ BET LOGGED
```

The exact:
- matchup
- market
- pick
- line
- odds
- sportsbook
- model edge at the time
- model probability
- break-even probability
- placement time

is persisted locally.

Current tracking assumes **1 unit per tapped bet**.

## Results

"My Tracking" shows only manually logged bets.

Unclicked model recommendations are excluded from:
- record
- units
- ROI
- market split
- pending count

Once refreshed model data includes the final game result, a logged spread/total is graded automatically.

## Install

Upload the ZIP into:

```text
/workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo/
```

Then:

```bash
cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo
unzip -o nfl_app_v4_8_track_only_my_bets.zip

cd mobile
npx expo install @react-native-async-storage/async-storage
npx expo start --tunnel -c
```

If AsyncStorage is already installed from v4.7, Expo will simply confirm the compatible version.
