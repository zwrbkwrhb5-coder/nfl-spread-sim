# NFL SIM v4.9.1 — Projected Moneyline Winner

This patch adds a clear **Projected Moneyline Winner** to every game using the existing model margin.

- Dashboard shows the projected winner and projected margin.
- Game Detail shows a larger projected-winner card.
- If moneyline odds are present, the card also shows the best available moneyline quote.
- If moneyline odds are not present, the projection still displays instead of disappearing.
- The iPhone safe-area back-button fix remains included.

## Install

Upload the ZIP into your Codespace project root, then run:

```bash
cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo
unzip -o nfl_app_v4_9_1_projected_moneyline_winner.zip

cd mobile
npx expo start --tunnel -c
```

No model re-export is required just to see the projected winner. Re-export with v4.9 later when real moneyline odds/candidates are available.
