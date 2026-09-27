from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

NFLVERSE_GAMES_URL = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)


def load_games() -> pd.DataFrame:
    print("Loading nflverse games.csv...")
    return pd.read_csv(NFLVERSE_GAMES_URL)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def normalize_context(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()

    keep = [
        "game_id","season","week","home_team","away_team",
        "home_rest","away_rest","div_game","roof","surface","temp","wind",
    ]
    for c in keep:
        if c not in g.columns:
            g[c] = np.nan

    out = g[keep].copy()

    out["home_rest"] = _num(out["home_rest"])
    out["away_rest"] = _num(out["away_rest"])
    out["rest_diff"] = out["home_rest"] - out["away_rest"]

    out["home_short_week"] = (out["home_rest"] <= 6).astype(float)
    out["away_short_week"] = (out["away_rest"] <= 6).astype(float)
    out["short_week_diff"] = out["away_short_week"] - out["home_short_week"]

    out["home_long_rest"] = (out["home_rest"] >= 10).astype(float)
    out["away_long_rest"] = (out["away_rest"] >= 10).astype(float)
    out["long_rest_diff"] = out["home_long_rest"] - out["away_long_rest"]

    div = out["div_game"].astype(str).str.lower()
    out["divisional_game"] = div.isin(["1","true","t","yes"]).astype(float)

    roof = out["roof"].astype(str).str.lower()
    out["indoor"] = roof.str.contains("dome|closed|indoors").astype(float)
    out["outdoor"] = 1.0 - out["indoor"]

    surface = out["surface"].astype(str).str.lower()
    out["grass"] = surface.str.contains("grass").astype(float)
    out["turf"] = surface.str.contains("turf|artificial").astype(float)

    out["temp_f"] = _num(out["temp"])
    out["wind_mph"] = _num(out["wind"])

    out["outdoor_temp_f"] = np.where(out["outdoor"] > 0, out["temp_f"], 70.0)
    out["outdoor_wind_mph"] = np.where(out["outdoor"] > 0, out["wind_mph"], 0.0)

    out["cold_game"] = ((out["outdoor"] > 0) & (out["temp_f"] <= 40)).astype(float)
    out["very_cold_game"] = ((out["outdoor"] > 0) & (out["temp_f"] <= 25)).astype(float)
    out["windy_game"] = ((out["outdoor"] > 0) & (out["wind_mph"] >= 15)).astype(float)
    out["very_windy_game"] = ((out["outdoor"] > 0) & (out["wind_mph"] >= 20)).astype(float)

    out["cold_degrees"] = np.where(
        out["outdoor"] > 0,
        np.maximum(0.0, 50.0 - out["temp_f"].fillna(50.0)),
        0.0,
    )
    out["wind_over_10"] = np.where(
        out["outdoor"] > 0,
        np.maximum(0.0, out["wind_mph"].fillna(0.0) - 10.0),
        0.0,
    )

    cols = [
        "game_id","season","week","home_team","away_team",
        "home_rest","away_rest","rest_diff",
        "home_short_week","away_short_week","short_week_diff",
        "home_long_rest","away_long_rest","long_rest_diff",
        "divisional_game","indoor","outdoor","grass","turf",
        "temp_f","wind_mph","outdoor_temp_f","outdoor_wind_mph",
        "cold_game","very_cold_game","windy_game","very_windy_game",
        "cold_degrees","wind_over_10",
    ]
    return out[cols].copy()


def merge_context(training: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    return training.merge(
        context,
        on=["game_id","season","week","home_team","away_team"],
        how="left",
        validate="one_to_one",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", default="artifacts/dual_market/training_games_qb_v17.csv")
    ap.add_argument("--output", default="artifacts/dual_market/training_games_qb_context_v24.csv")
    ap.add_argument("--context-output", default="artifacts/dual_market/context_features_v24.csv")
    args = ap.parse_args()

    training = pd.read_csv(args.training_csv)
    games = load_games()
    context = normalize_context(games)
    merged = merge_context(training, context)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    context.to_csv(args.context_output, index=False)

    print(f"Saved training table: {args.output} ({len(merged):,} rows)")
    print(f"Saved context table:  {args.context_output} ({len(context):,} rows)")
    print(f"Rows with matched context: {merged['rest_diff'].notna().sum():,} / {len(merged):,}")


if __name__ == "__main__":
    main()
