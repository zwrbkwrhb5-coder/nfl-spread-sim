from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

NFLVERSE_GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

REQUIRED_SOURCE = {
    "game_id","season","game_type","week",
    "away_team","away_score","home_team","home_score",
    "spread_line","away_spread_odds","home_spread_odds",
    "total_line","under_odds","over_odds",
}

def load_games(source: str = NFLVERSE_GAMES_URL) -> pd.DataFrame:
    df = pd.read_csv(source, low_memory=False)
    missing = REQUIRED_SOURCE - set(df.columns)
    if missing:
        raise ValueError(f"nflverse source missing expected columns: {sorted(missing)}")
    return df

def _fill_price(series: pd.Series, default_odds: float) -> tuple[pd.Series, pd.Series]:
    s = pd.to_numeric(series, errors="coerce")
    imputed = s.isna()
    return s.fillna(float(default_odds)), imputed

def build_historical_market(
    games: pd.DataFrame,
    start_season: int,
    end_season: int,
    default_odds: float = -110.0,
    include_postseason: bool = True,
) -> pd.DataFrame:
    d = games[
        games["season"].between(start_season, end_season, inclusive="both")
    ].copy()

    allowed_types = ["REG","WC","DIV","CON","SB"] if include_postseason else ["REG"]
    d = d[d["game_type"].isin(allowed_types)].copy()

    numeric_cols = [
        "away_score","home_score","spread_line",
        "away_spread_odds","home_spread_odds",
        "total_line","under_odds","over_odds",
    ]
    for c in numeric_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=[
        "game_id","season","away_team","home_team",
        "away_score","home_score","spread_line","total_line"
    ]).copy()

    # nflverse convention:
    # spread_line > 0 => home team favored by that many points.
    #
    # Our backtester convention:
    # home_spread = standard sportsbook notation for the home side,
    # e.g. home favored by 3.5 => -3.5.
    d["home_spread"] = -d["spread_line"].astype(float)
    d["away_spread"] = d["spread_line"].astype(float)

    d["home_spread_odds"], d["home_spread_odds_imputed"] = _fill_price(
        d["home_spread_odds"], default_odds
    )
    d["away_spread_odds"], d["away_spread_odds_imputed"] = _fill_price(
        d["away_spread_odds"], default_odds
    )
    d["over_odds"], d["over_odds_imputed"] = _fill_price(
        d["over_odds"], default_odds
    )
    d["under_odds"], d["under_odds_imputed"] = _fill_price(
        d["under_odds"], default_odds
    )

    # nflverse games.csv documents spread_line / total_line as closing lines.
    d["closing_home_spread"] = d["home_spread"]
    d["closing_total"] = d["total_line"].astype(float)

    d["actual_margin"] = d["home_score"] - d["away_score"]
    d["actual_total"] = d["home_score"] + d["away_score"]

    cols = [
        "game_id","season","game_type","week","gameday",
        "home_team","away_team","home_score","away_score",
        "home_spread","away_spread",
        "home_spread_odds","away_spread_odds",
        "total_line","over_odds","under_odds",
        "closing_home_spread","closing_total",
        "actual_margin","actual_total",
        "home_spread_odds_imputed","away_spread_odds_imputed",
        "over_odds_imputed","under_odds_imputed",
    ]
    cols = [c for c in cols if c in d.columns]
    out = d[cols].sort_values(["season","week","game_id"]).reset_index(drop=True)
    return out

def summarize(df: pd.DataFrame) -> dict:
    return {
        "games": int(len(df)),
        "season_min": int(df["season"].min()) if len(df) else None,
        "season_max": int(df["season"].max()) if len(df) else None,
        "home_spread_odds_imputed": int(df["home_spread_odds_imputed"].sum()),
        "away_spread_odds_imputed": int(df["away_spread_odds_imputed"].sum()),
        "over_odds_imputed": int(df["over_odds_imputed"].sum()),
        "under_odds_imputed": int(df["under_odds_imputed"].sum()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-season", type=int, default=2018)
    ap.add_argument("--end-season", type=int, default=2025)
    ap.add_argument("--output", default="data/historical_market.csv")
    ap.add_argument("--default-odds", type=float, default=-110.0)
    ap.add_argument("--regular-season-only", action="store_true")
    ap.add_argument("--source", default=NFLVERSE_GAMES_URL)
    args = ap.parse_args()

    games = load_games(args.source)
    out = build_historical_market(
        games,
        args.start_season,
        args.end_season,
        default_odds=args.default_odds,
        include_postseason=not args.regular_season_only,
    )

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    print(f"Saved {len(out):,} games to {path}")
    print(json.dumps(summarize(out), indent=2))

if __name__ == "__main__":
    import json
    main()
