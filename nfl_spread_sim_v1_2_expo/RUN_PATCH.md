# v2.9.1 Team Games Patch

Run from the project root:

```bash
unzip -n nfl_spread_totals_v2_9_1_team_games_patch.zip
source .venv-model/bin/activate
python -m score_model.rebuild_team_games_v29
ls -lh artifacts/team_games.parquet
```

Then run v2.9 normally.
