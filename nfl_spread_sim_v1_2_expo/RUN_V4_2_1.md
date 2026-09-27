# v4.2.1 Expo Router fix

This fixes:

```text
[expo-router]: You are passing an array of styles to a child of <Slot>.
Consider flattening the styles with StyleSheet.flatten...
```

The problem was caused by `Link asChild` wrapping `Pressable` elements whose
`style` prop was an array.

## Install from the repository root

```bash
cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo
unzip -o nfl_app_v4_2_1_expo_router_fix.zip
```

Then restart Expo:

```bash
cd mobile
npx expo start --tunnel -c
```

No npm install is required.
