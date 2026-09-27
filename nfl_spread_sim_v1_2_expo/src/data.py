from __future__ import annotations
from pathlib import Path
import pandas as pd

PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp/play_by_play_{season}.parquet"
)

# Columns used by team model + QB model.
KEEP = [
    "game_id", "season", "week", "season_type", "home_team", "away_team",
    "posteam", "defteam", "epa", "success", "pass", "rush",
    "complete_pass", "yards_gained", "sack", "interception",
    "fumble_lost", "posteam_score", "defteam_score",
    "home_score", "away_score", "play_type", "qb_kneel", "qb_spike",
    "passer_player_id", "passer_player_name", "cpoe", "air_yards", "drive"
]

def load_pbp(season: int, cache_dir: str = "data/raw") -> pd.DataFrame:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"play_by_play_{season}.parquet"

    if not path.exists():
        # Read all columns first so model remains robust if source schemas vary.
        df = pd.read_parquet(PBP_URL.format(season=season))
        cols = [c for c in KEEP if c in df.columns]
        df = df[cols]
        df.to_parquet(path, index=False)
        return df

    df = pd.read_parquet(path)
    cols = [c for c in KEEP if c in df.columns]
    return df[cols]

def load_seasons(start: int, end: int, cache_dir: str = "data/raw") -> pd.DataFrame:
    frames = [load_pbp(y, cache_dir) for y in range(start, end + 1)]
    return pd.concat(frames, ignore_index=True)
