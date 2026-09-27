from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .injury_layer_v19 import (
    normalize_injuries,
    aggregate_team_week,
    merge_into_training,
    load_seasons,
)

NFLVERSE_ROSTERS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "rosters/roster_{season}.parquet"
)

NFLVERSE_WEEKLY_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "player_stats/player_stats.csv"
)


POSITION_BASE = {
    "QB": 3.0,
    "OT": 1.5, "OL": 1.35, "OG": 1.2, "C": 1.25,
    "WR": 1.1, "TE": 0.95, "RB": 0.75,
    "EDGE": 1.2, "DE": 1.1, "DT": 0.95, "DL": 1.0,
    "LB": 0.95, "CB": 1.15, "S": 1.0, "DB": 1.05,
    "K": 0.4, "P": 0.2,
}


def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def load_rosters(start_season: int, end_season: int) -> pd.DataFrame:
    frames = []
    for season in range(start_season, end_season + 1):
        url = NFLVERSE_ROSTERS_URL.format(season=season)
        try:
            print(f"Loading roster {season}...")
            frames.append(pd.read_parquet(url))
        except Exception as e:
            print(f"WARNING roster {season}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_rosters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    player_col = first_existing(df.columns, ["gsis_id","player_id","espn_id"])
    team_col = first_existing(df.columns, ["team","team_abbr","club"])
    pos_col = first_existing(df.columns, ["position","pos"])
    season_col = first_existing(df.columns, ["season"])
    status_col = first_existing(df.columns, ["status","roster_status"])
    depth_col = first_existing(df.columns, ["depth_chart_position","depth_position","depth"])

    out = pd.DataFrame({
        "player_key": df[player_col].astype(str) if player_col else df.index.astype(str),
        "team": df[team_col].astype(str) if team_col else "",
        "position": df[pos_col].astype(str).str.upper() if pos_col else "",
        "season": pd.to_numeric(df[season_col], errors="coerce") if season_col else np.nan,
        "roster_status": df[status_col].astype(str) if status_col else "",
        "depth": df[depth_col].astype(str) if depth_col else "",
    })
    return out


def starter_score_from_depth(depth: str) -> float:
    s = str(depth).strip().lower()
    if not s:
        return 0.55
    if any(x in s for x in ["starter","1","first"]):
        return 1.0
    if any(x in s for x in ["2","second"]):
        return 0.65
    if any(x in s for x in ["3","third"]):
        return 0.4
    return 0.55


def add_player_value(inj: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    d = inj.copy()

    if not rosters.empty:
        r = normalize_rosters(rosters)
        d = d.merge(
            r[["player_key","season","team","depth","roster_status"]],
            on=["player_key","season","team"],
            how="left",
        )
    else:
        d["depth"] = ""
        d["roster_status"] = ""

    d["starter_weight"] = d["depth"].map(starter_score_from_depth).fillna(0.55)

    # Position baseline from normalized injury position.
    d["position_value"] = d["position"].map(POSITION_BASE).fillna(0.8)

    # Current status severity already in injury_impact via v1.9.
    # Decompose it so player value can modulate it.
    denom = d["position_weight"].replace(0, np.nan)
    d["status_component"] = (d["injury_impact"] / denom).fillna(d["status_weight"])

    # Player-value injury signal:
    # status severity x position value x starter/depth importance.
    d["player_value_injury_impact"] = (
        d["status_component"] *
        d["position_value"] *
        d["starter_weight"]
    )

    d["qb_player_value_impact"] = np.where(
        d["position"] == "QB", d["player_value_injury_impact"], 0.0
    )
    d["ol_player_value_impact"] = np.where(
        d["position"].isin(["OL","OT","OG","T","G","C"]),
        d["player_value_injury_impact"], 0.0
    )
    d["secondary_player_value_impact"] = np.where(
        d["position"].isin(["CB","S","DB"]),
        d["player_value_injury_impact"], 0.0
    )
    d["skill_player_value_impact"] = np.where(
        d["position"].isin(["WR","TE","RB"]),
        d["player_value_injury_impact"], 0.0
    )

    return d


def aggregate_player_value_team_week(d: pd.DataFrame) -> pd.DataFrame:
    d = (
        d.sort_values("player_value_injury_impact", ascending=False)
        .drop_duplicates(["season","week","team","player_key"])
    )

    return d.groupby(["season","week","team"], as_index=False).agg(
        pv_total_injury_impact=("player_value_injury_impact","sum"),
        pv_qb_injury_impact=("qb_player_value_impact","sum"),
        pv_ol_injury_impact=("ol_player_value_impact","sum"),
        pv_secondary_injury_impact=("secondary_player_value_impact","sum"),
        pv_skill_injury_impact=("skill_player_value_impact","sum"),
        pv_injured_players=("player_key","nunique"),
    )


def merge_player_value(training: pd.DataFrame, team_week: pd.DataFrame) -> pd.DataFrame:
    t = training.copy()

    home = team_week.rename(columns={
        "team":"home_team",
        "pv_total_injury_impact":"home_pv_total_injury_impact",
        "pv_qb_injury_impact":"home_pv_qb_injury_impact",
        "pv_ol_injury_impact":"home_pv_ol_injury_impact",
        "pv_secondary_injury_impact":"home_pv_secondary_injury_impact",
        "pv_skill_injury_impact":"home_pv_skill_injury_impact",
        "pv_injured_players":"home_pv_injured_players",
    })
    away = team_week.rename(columns={
        "team":"away_team",
        "pv_total_injury_impact":"away_pv_total_injury_impact",
        "pv_qb_injury_impact":"away_pv_qb_injury_impact",
        "pv_ol_injury_impact":"away_pv_ol_injury_impact",
        "pv_secondary_injury_impact":"away_pv_secondary_injury_impact",
        "pv_skill_injury_impact":"away_pv_skill_injury_impact",
        "pv_injured_players":"away_pv_injured_players",
    })

    t = t.merge(home, on=["season","week","home_team"], how="left")
    t = t.merge(away, on=["season","week","away_team"], how="left")

    bases = [
        "pv_total_injury_impact",
        "pv_qb_injury_impact",
        "pv_ol_injury_impact",
        "pv_secondary_injury_impact",
        "pv_skill_injury_impact",
        "pv_injured_players",
    ]
    for b in bases:
        t[f"home_{b}"] = t[f"home_{b}"].fillna(0)
        t[f"away_{b}"] = t[f"away_{b}"].fillna(0)
        t[f"diff_{b}"] = t[f"away_{b}"] - t[f"home_{b}"]
        t[f"sum_{b}"] = t[f"away_{b}"] + t[f"home_{b}"]

    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", default="artifacts/dual_market/training_games_qb_injury_v19.csv")
    ap.add_argument("--start-season", type=int, default=2018)
    ap.add_argument("--end-season", type=int, default=2024)
    ap.add_argument("--injury-csv", default=None)
    ap.add_argument("--output", default="artifacts/dual_market/training_games_qb_injury_pv_v20.csv")
    ap.add_argument("--team-week-output", default="artifacts/dual_market/player_value_injury_team_week_v20.csv")
    args = ap.parse_args()

    training = pd.read_csv(args.training_csv)

    if args.injury_csv:
        raw = pd.read_csv(args.injury_csv)
    else:
        raw, _ = load_seasons(args.start_season, args.end_season)

    inj = normalize_injuries(raw)
    rosters = load_rosters(args.start_season, args.end_season)
    inj = add_player_value(inj, rosters)
    tw = aggregate_player_value_team_week(inj)
    merged = merge_player_value(training, tw)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    tw.to_csv(args.team_week_output, index=False)

    print(f"Saved {args.output} ({len(merged):,} rows)")
    print(f"Saved {args.team_week_output} ({len(tw):,} team-weeks)")


if __name__ == "__main__":
    main()
