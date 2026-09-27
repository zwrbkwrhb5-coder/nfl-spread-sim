# NFL SIM v5.0.1 - Week 4 filename recovery

The v3.1 live runner honors `--week 4` for modeling but currently writes its output to fixed Week 3 filenames. The original v5.0 combiner therefore could not find the expected Week 4 files.

## Recover the Week 4 model run you already completed

```bash
cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo
source .venv-model/bin/activate
bash score_model/recover_week4_v501.sh
```

This does not call The Odds API again. It validates that the mislabeled files contain `2026_04_` game IDs, copies them to the proper Week 4 filenames, preserves Week 3 from the existing mobile snapshot, and creates the two-week app export.

Then:

```bash
cd mobile
npx expo start --tunnel -c
```

## Future Week 4 refreshes

Use this corrected command instead of the original v5.0 script:

```bash
cd /workspaces/nfl-spread-sim/nfl_spread_sim_v1_2_expo
source .venv-model/bin/activate
bash score_model/refresh_week4_v501.sh
```
