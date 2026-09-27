from pathlib import Path
import gc

import pandas as pd

from src.features import build_team_games


PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp/play_by_play_{season}.parquet"
)

START_SEASON = 2018
END_SEASON = 2025


def main():
    parts = []

    for season in range(START_SEASON, END_SEASON + 1):
        print(f"\nLoading {season} PBP...")

        pbp = pd.read_parquet(
            PBP_URL.format(season=season)
        )

        print(
            f"  {season}: {len(pbp):,} plays loaded"
        )

        team_games = build_team_games(pbp)

        print(
            f"  {season}: {len(team_games):,} team-game rows built"
        )

        parts.append(team_games.copy())

        del pbp
        del team_games
        gc.collect()

    print("\nCombining seasons...")

    out = pd.concat(
        parts,
        ignore_index=True,
    )

    out = (
        out
        .drop_duplicates(
            ["game_id", "team"],
            keep="last",
        )
        .sort_values(
            ["season", "week", "game_id", "team"]
        )
        .reset_index(drop=True)
    )

    path = Path("artifacts/team_games.parquet")
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_parquet(
        path,
        index=False,
    )

    print("\nDONE")
    print(f"Saved: {path}")
    print(f"Rows: {len(out):,}")
    print(
        f"Seasons: {out['season'].min()}-{out['season'].max()}"
    )


if __name__ == "__main__":
    main()
