from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

NFLVERSE_PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp/play_by_play_{season}.parquet"
)


def _first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def load_pbp(start_season: int, end_season: int) -> pd.DataFrame:
    frames = []
    for season in range(start_season, end_season + 1):
        url = NFLVERSE_PBP_URL.format(season=season)
        print(f"Loading PBP {season}...")
        frames.append(pd.read_parquet(url))
    return pd.concat(frames, ignore_index=True)


def derive_opening_drives(pbp: pd.DataFrame) -> pd.DataFrame:
    d = pbp.copy()

    required = ["game_id","season","week","posteam","defteam"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required PBP columns: {missing}")

    drive_col = _first_existing(d.columns, ["drive","drive_id"])
    if drive_col is None:
        raise ValueError("Could not find drive column.")

    # Limit to real offensive plays when possible.
    play_type = d.get("play_type")
    if play_type is not None:
        d = d[~d["play_type"].isin(["no_play","qb_kneel","qb_spike"])].copy()

    d = d[d["posteam"].notna() & d["defteam"].notna()].copy()
    d["_drive"] = d[drive_col]

    # First offensive drive for each team in a game.
    first_drive = (
        d.groupby(["game_id","posteam"])["_drive"]
        .min()
        .rename("opening_drive_id")
        .reset_index()
    )
    d = d.merge(first_drive, on=["game_id","posteam"], how="inner")
    od = d[d["_drive"] == d["opening_drive_id"]].copy()

    # Points scored on drive. Prefer drive result if available; fallback to score delta.
    result_col = _first_existing(od.columns, ["drive_result","fixed_drive_result"])
    if result_col:
        result = od.groupby(["game_id","season","week","posteam","defteam"])[result_col].last().reset_index()
        map_points = {
            "Touchdown": 7.0, "Field goal": 3.0, "Field Goal": 3.0,
            "Punt": 0.0, "Interception": 0.0, "Fumble": 0.0,
            "Turnover on downs": 0.0, "Downs": 0.0, "End of half": 0.0,
            "Missed field goal": 0.0, "Missed Field Goal": 0.0,
        }
        result["opening_points"] = result[result_col].map(map_points).fillna(0.0)
        drive_result = result[["game_id","season","week","posteam","defteam","opening_points"]]
    else:
        # Fallback using posteam_score change through drive.
        score_col = _first_existing(od.columns, ["posteam_score"])
        if score_col is None:
            drive_result = (
                od.groupby(["game_id","season","week","posteam","defteam"], as_index=False)
                .size()
            )
            drive_result["opening_points"] = 0.0
            drive_result = drive_result.drop(columns=["size"])
        else:
            g = od.groupby(["game_id","season","week","posteam","defteam"])
            tmp = g[score_col].agg(["first","last"]).reset_index()
            tmp["opening_points"] = (tmp["last"] - tmp["first"]).clip(lower=0)
            drive_result = tmp[["game_id","season","week","posteam","defteam","opening_points"]]

    agg = od.groupby(
        ["game_id","season","week","posteam","defteam"], as_index=False
    ).agg(
        opening_epa=("epa","sum"),
        opening_success_rate=("success","mean"),
        opening_turnover=("turnover", "max") if "turnover" in od.columns else ("epa","size"),
        opening_plays=("epa","size"),
    )

    if "turnover" not in od.columns:
        # Approximate turnover with interception/fumble_lost.
        ints = pd.to_numeric(od.get("interception", 0), errors="coerce").fillna(0)
        fum = pd.to_numeric(od.get("fumble_lost", 0), errors="coerce").fillna(0)
        od["_to"] = ((ints > 0) | (fum > 0)).astype(int)
        tos = od.groupby(["game_id","season","week","posteam","defteam"])["_to"].max().reset_index(name="opening_turnover")
        agg = agg.drop(columns=["opening_turnover"]).merge(
            tos, on=["game_id","season","week","posteam","defteam"], how="left"
        )

    agg = agg.merge(
        drive_result,
        on=["game_id","season","week","posteam","defteam"],
        how="left",
    )

    agg["opening_scored"] = (agg["opening_points"] > 0).astype(float)
    agg["opening_td"] = (agg["opening_points"] >= 6).astype(float)
    agg["opening_fg"] = ((agg["opening_points"] >= 3) & (agg["opening_points"] < 6)).astype(float)

    return agg.rename(columns={"posteam":"team","defteam":"opponent"})


def add_pregame_tendencies(drives: pd.DataFrame, halflife_games: float = 6.0, min_games: int = 3) -> pd.DataFrame:
    d = drives.sort_values(["team","season","week","game_id"]).copy()

    off_cols = [
        "opening_points","opening_epa","opening_success_rate",
        "opening_turnover","opening_scored","opening_td","opening_fg",
    ]

    out = []
    for team, g in d.groupby("team", sort=False):
        g = g.copy()
        for c in off_cols:
            g[f"pre_off_{c}"] = (
                g[c].shift(1)
                .ewm(halflife=halflife_games, adjust=False, min_periods=min_games)
                .mean()
            )
        out.append(g)

    off = pd.concat(out, ignore_index=True)

    # Defensive tendencies: what opponents did on their opening drive.
    dd = drives.rename(columns={"opponent":"team","team":"opponent"}).sort_values(
        ["team","season","week","game_id"]
    ).copy()

    dout = []
    for team, g in dd.groupby("team", sort=False):
        g = g.copy()
        for c in off_cols:
            g[f"pre_def_allowed_{c}"] = (
                g[c].shift(1)
                .ewm(halflife=halflife_games, adjust=False, min_periods=min_games)
                .mean()
            )
        dout.append(g)
    deff = pd.concat(dout, ignore_index=True)

    keep_off = [
        "game_id","season","week","team"
    ] + [f"pre_off_{c}" for c in off_cols]
    keep_def = [
        "game_id","season","week","team"
    ] + [f"pre_def_allowed_{c}" for c in off_cols]

    return off[keep_off].merge(
        deff[keep_def],
        on=["game_id","season","week","team"],
        how="outer",
    )


def merge_into_training(training: pd.DataFrame, tendencies: pd.DataFrame) -> pd.DataFrame:
    t = training.copy()

    home = tendencies.rename(columns={
        "team":"home_team",
        **{c:f"home_{c}" for c in tendencies.columns if c.startswith("pre_")}
    })
    away = tendencies.rename(columns={
        "team":"away_team",
        **{c:f"away_{c}" for c in tendencies.columns if c.startswith("pre_")}
    })

    t = t.merge(home, on=["game_id","season","week","home_team"], how="left")
    t = t.merge(away, on=["game_id","season","week","away_team"], how="left")

    # Matchup interactions: offense tendency versus opposing defense allowed tendency.
    metrics = [
        "opening_points","opening_epa","opening_success_rate",
        "opening_turnover","opening_scored","opening_td","opening_fg",
    ]

    for m in metrics:
        t[f"home_opening_matchup_{m}"] = (
            t[f"home_pre_off_{m}"] + t[f"away_pre_def_allowed_{m}"]
        ) / 2.0
        t[f"away_opening_matchup_{m}"] = (
            t[f"away_pre_off_{m}"] + t[f"home_pre_def_allowed_{m}"]
        ) / 2.0

        t[f"diff_opening_matchup_{m}"] = (
            t[f"home_opening_matchup_{m}"] - t[f"away_opening_matchup_{m}"]
        )
        t[f"sum_opening_matchup_{m}"] = (
            t[f"home_opening_matchup_{m}"] + t[f"away_opening_matchup_{m}"]
        )

    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", default="artifacts/dual_market/training_games_qb_v17.csv")
    ap.add_argument("--start-season", type=int, default=2018)
    ap.add_argument("--end-season", type=int, default=2025)
    ap.add_argument("--halflife-games", type=float, default=6.0)
    ap.add_argument("--min-games", type=int, default=3)
    ap.add_argument("--output", default="artifacts/dual_market/training_games_qb_opening_v22.csv")
    ap.add_argument("--drive-output", default="artifacts/dual_market/opening_drives_v22.csv")
    args = ap.parse_args()

    training = pd.read_csv(args.training_csv)
    pbp = load_pbp(args.start_season, args.end_season)
    drives = derive_opening_drives(pbp)
    tendencies = add_pregame_tendencies(
        drives,
        halflife_games=args.halflife_games,
        min_games=args.min_games,
    )
    merged = merge_into_training(training, tendencies)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    drives.to_csv(args.drive_output, index=False)

    print(f"Saved: {args.output} ({len(merged):,} rows)")
    print(f"Saved: {args.drive_output} ({len(drives):,} team-games)")


if __name__ == "__main__":
    main()
