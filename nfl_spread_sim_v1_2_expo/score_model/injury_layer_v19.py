from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


NFLVERSE_INJURY_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "injuries/injuries_{season}.parquet"
)

STATUS_WEIGHT = {
    "out": 1.00,
    "doubtful": 0.75,
    "questionable": 0.35,
    "probable": 0.10,
    "ir": 1.00,
    "pup": 1.00,
    "reserve": 1.00,
}

POSITION_WEIGHT = {
    "QB": 3.0,
    "OL": 1.35,
    "OT": 1.50,
    "T": 1.50,
    "OG": 1.20,
    "G": 1.20,
    "C": 1.25,
    "WR": 1.10,
    "TE": 0.95,
    "RB": 0.75,
    "DL": 1.00,
    "DE": 1.10,
    "DT": 0.95,
    "EDGE": 1.20,
    "LB": 0.95,
    "CB": 1.15,
    "S": 1.00,
    "DB": 1.05,
    "K": 0.40,
    "P": 0.20,
}


def norm_status(x):
    s = str(x).strip().lower()
    for k in STATUS_WEIGHT:
        if k in s:
            return k
    return s


def norm_pos(x):
    s = str(x).strip().upper()
    if s in {"LT","RT"}:
        return "OT"
    if s in {"LG","RG"}:
        return "OG"
    return s


def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def normalize_injuries(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    team_col = first_existing(d.columns, ["team", "team_abbr", "club"])
    week_col = first_existing(d.columns, ["week", "game_week"])
    season_col = first_existing(d.columns, ["season"])
    pos_col = first_existing(d.columns, ["position", "pos"])
    status_col = first_existing(d.columns, ["report_status", "status", "practice_status"])
    player_col = first_existing(d.columns, ["gsis_id", "player_id", "full_name", "player_name"])

    required = {
        "team": team_col, "week": week_col, "season": season_col,
        "position": pos_col, "status": status_col
    }
    missing = [k for k,v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing required injury columns: {missing}")

    out = pd.DataFrame({
        "season": pd.to_numeric(d[season_col], errors="coerce"),
        "week": pd.to_numeric(d[week_col], errors="coerce"),
        "team": d[team_col].astype(str),
        "position": d[pos_col].map(norm_pos),
        "status": d[status_col].map(norm_status),
        "player_key": d[player_col].astype(str) if player_col else d.index.astype(str),
    })

    out["status_weight"] = out["status"].map(STATUS_WEIGHT).fillna(0.15)
    out["position_weight"] = out["position"].map(POSITION_WEIGHT).fillna(0.80)
    out["injury_impact"] = out["status_weight"] * out["position_weight"]

    return out.dropna(subset=["season","week","team"]).copy()


def aggregate_team_week(inj: pd.DataFrame) -> pd.DataFrame:
    d = inj.copy()

    d["qb_impact"] = np.where(d["position"] == "QB", d["injury_impact"], 0.0)
    d["ol_impact"] = np.where(
        d["position"].isin(["OL","OT","OG","T","G","C"]),
        d["injury_impact"], 0.0
    )
    d["secondary_impact"] = np.where(
        d["position"].isin(["CB","S","DB"]),
        d["injury_impact"], 0.0
    )
    d["skill_impact"] = np.where(
        d["position"].isin(["WR","TE","RB"]),
        d["injury_impact"], 0.0
    )

    # Deduplicate player/team/week if source has multiple entries.
    d = (
        d.sort_values("injury_impact", ascending=False)
        .drop_duplicates(["season","week","team","player_key"])
    )

    g = d.groupby(["season","week","team"], as_index=False).agg(
        total_injury_impact=("injury_impact","sum"),
        qb_injury_impact=("qb_impact","sum"),
        ol_injury_impact=("ol_impact","sum"),
        secondary_injury_impact=("secondary_impact","sum"),
        skill_injury_impact=("skill_impact","sum"),
        injured_players=("player_key","nunique"),
    )
    return g


def merge_into_training(training: pd.DataFrame, team_week: pd.DataFrame) -> pd.DataFrame:
    t = training.copy()

    home = team_week.rename(columns={
        "team":"home_team",
        "total_injury_impact":"home_total_injury_impact",
        "qb_injury_impact":"home_qb_injury_impact",
        "ol_injury_impact":"home_ol_injury_impact",
        "secondary_injury_impact":"home_secondary_injury_impact",
        "skill_injury_impact":"home_skill_injury_impact",
        "injured_players":"home_injured_players",
    })
    away = team_week.rename(columns={
        "team":"away_team",
        "total_injury_impact":"away_total_injury_impact",
        "qb_injury_impact":"away_qb_injury_impact",
        "ol_injury_impact":"away_ol_injury_impact",
        "secondary_injury_impact":"away_secondary_injury_impact",
        "skill_injury_impact":"away_skill_injury_impact",
        "injured_players":"away_injured_players",
    })

    t = t.merge(home, on=["season","week","home_team"], how="left")
    t = t.merge(away, on=["season","week","away_team"], how="left")

    impact_bases = [
        "total_injury_impact",
        "qb_injury_impact",
        "ol_injury_impact",
        "secondary_injury_impact",
        "skill_injury_impact",
        "injured_players",
    ]
    for b in impact_bases:
        t[f"home_{b}"] = t[f"home_{b}"].fillna(0)
        t[f"away_{b}"] = t[f"away_{b}"].fillna(0)
        # Positive means away is more injured -> relative advantage home.
        t[f"diff_{b}"] = t[f"away_{b}"] - t[f"home_{b}"]
        # Totals may care about aggregate game injury environment.
        t[f"sum_{b}"] = t[f"away_{b}"] + t[f"home_{b}"]

    return t


def load_seasons(start_season, end_season):
    frames = []
    failures = []
    for season in range(start_season, end_season + 1):
        url = NFLVERSE_INJURY_URL.format(season=season)
        try:
            print(f"Loading injuries {season}...")
            frames.append(pd.read_parquet(url))
        except Exception as e:
            failures.append((season, str(e)))
            print(f"WARNING: could not load {season}: {e}")
    if not frames:
        raise RuntimeError("No injury seasons could be loaded.")
    return pd.concat(frames, ignore_index=True), failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", default="artifacts/dual_market/training_games_qb_v17.csv")
    ap.add_argument("--start-season", type=int, default=2018)
    ap.add_argument("--end-season", type=int, default=2024)
    ap.add_argument("--injury-csv", default=None)
    ap.add_argument("--output", default="artifacts/dual_market/training_games_qb_injury_v19.csv")
    ap.add_argument("--team-week-output", default="artifacts/dual_market/injury_team_week_v19.csv")
    args = ap.parse_args()

    training = pd.read_csv(args.training_csv)

    if args.injury_csv:
        raw = pd.read_csv(args.injury_csv)
        failures = []
    else:
        raw, failures = load_seasons(args.start_season, args.end_season)

    norm = normalize_injuries(raw)
    tw = aggregate_team_week(norm)
    merged = merge_into_training(training, tw)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    tw.to_csv(args.team_week_output, index=False)

    print(f"Saved: {args.output} ({len(merged):,} rows)")
    print(f"Saved: {args.team_week_output} ({len(tw):,} team-weeks)")
    if failures:
        print("Failed seasons:")
        for season, msg in failures:
            print(f"  {season}: {msg}")


if __name__ == "__main__":
    main()
