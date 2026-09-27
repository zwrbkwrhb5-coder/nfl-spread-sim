from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


NFLVERSE_PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp/play_by_play_{season}.parquet"
)

QB_RATE_COLS = {
    "qb_epa_per_dropback": ("epa", "mean"),
    "qb_success_rate": ("success", "mean"),
    "qb_cpoe": ("cpoe", "mean"),
    "qb_sack_rate": ("sack", "mean"),
    "qb_interception_rate": ("interception", "mean"),
    "qb_explosive_pass_rate": ("explosive_pass", "mean"),
}


def _first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def normalize_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    d = pbp.copy()

    passer_id = _first_existing(
        d.columns,
        ["passer_player_id", "passer_id", "passer"]
    )
    passer_name = _first_existing(
        d.columns,
        ["passer_player_name", "passer_name"]
    )

    if passer_id is None:
        raise ValueError("Could not find passer id column in PBP.")

    if "game_id" not in d or "posteam" not in d:
        raise ValueError("PBP must contain game_id and posteam.")

    if "season" not in d or "week" not in d:
        raise ValueError("PBP must contain season and week.")

    # NFL dropback approximation.
    pass_attempt = pd.to_numeric(d.get("pass_attempt", 0), errors="coerce").fillna(0)
    sack = pd.to_numeric(d.get("sack", 0), errors="coerce").fillna(0)
    scramble = pd.to_numeric(d.get("qb_scramble", 0), errors="coerce").fillna(0)

    d["_dropback"] = ((pass_attempt > 0) | (sack > 0) | (scramble > 0)).astype(int)
    d = d[(d["_dropback"] == 1) & d[passer_id].notna() & d["posteam"].notna()].copy()

    d["qb_id"] = d[passer_id].astype(str)
    d["qb_name"] = (
        d[passer_name].astype(str)
        if passer_name is not None
        else d["qb_id"]
    )

    d["epa"] = pd.to_numeric(d.get("epa", np.nan), errors="coerce")
    d["success"] = pd.to_numeric(d.get("success", np.nan), errors="coerce")
    d["cpoe"] = pd.to_numeric(d.get("cpoe", np.nan), errors="coerce")
    d["sack"] = sack.loc[d.index]
    d["interception"] = pd.to_numeric(d.get("interception", 0), errors="coerce").fillna(0)

    yards = pd.to_numeric(d.get("yards_gained", np.nan), errors="coerce")
    pass_att = pass_attempt.loc[d.index]
    d["explosive_pass"] = ((pass_att > 0) & (yards >= 20)).astype(float)

    return d


def build_qb_game_table(pbp: pd.DataFrame) -> pd.DataFrame:
    d = normalize_pbp(pbp)

    agg = (
        d.groupby(
            ["game_id", "season", "week", "posteam", "qb_id", "qb_name"],
            as_index=False,
        )
        .agg(
            dropbacks=("_dropback", "sum"),
            qb_epa_per_dropback=("epa", "mean"),
            qb_success_rate=("success", "mean"),
            qb_cpoe=("cpoe", "mean"),
            qb_sack_rate=("sack", "mean"),
            qb_interception_rate=("interception", "mean"),
            qb_explosive_pass_rate=("explosive_pass", "mean"),
        )
    )

    # Starter approximation: QB with most dropbacks for that team-game.
    agg = agg.sort_values(
        ["game_id", "posteam", "dropbacks"],
        ascending=[True, True, False],
    )
    starter = agg.drop_duplicates(["game_id", "posteam"]).copy()
    starter = starter.rename(columns={"posteam": "team"})
    return starter.reset_index(drop=True)


def add_pregame_qb_form(
    qb_games: pd.DataFrame,
    halflife_games: float = 5.0,
    min_games: int = 2,
) -> pd.DataFrame:
    d = qb_games.copy().sort_values(
        ["qb_id", "season", "week", "game_id"]
    ).reset_index(drop=True)

    metric_cols = [
        "qb_epa_per_dropback",
        "qb_success_rate",
        "qb_cpoe",
        "qb_sack_rate",
        "qb_interception_rate",
        "qb_explosive_pass_rate",
    ]

    out = []
    for qb_id, g in d.groupby("qb_id", sort=False):
        g = g.copy()

        g["qb_prior_starts"] = np.arange(len(g), dtype=int)
        g["qb_prior_dropbacks"] = g["dropbacks"].cumsum().shift(1).fillna(0)

        for c in metric_cols:
            shifted = g[c].shift(1)
            g[f"pre_{c}"] = (
                shifted.ewm(
                    halflife=halflife_games,
                    adjust=False,
                    min_periods=min_games,
                ).mean()
            )

        out.append(g)

    return pd.concat(out, ignore_index=True)


def build_game_qb_features(qb_form: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "game_id", "season", "week", "team", "qb_id", "qb_name",
        "qb_prior_starts", "qb_prior_dropbacks",
        "pre_qb_epa_per_dropback",
        "pre_qb_success_rate",
        "pre_qb_cpoe",
        "pre_qb_sack_rate",
        "pre_qb_interception_rate",
        "pre_qb_explosive_pass_rate",
    ]
    return qb_form[keep].copy()


def merge_qb_into_training(
    training: pd.DataFrame,
    qb_features: pd.DataFrame,
) -> pd.DataFrame:
    t = training.copy()
    q = qb_features.copy()

    home = q.rename(columns={
        "team": "home_team",
        "qb_id": "home_qb_id",
        "qb_name": "home_qb_name",
        "qb_prior_starts": "home_qb_prior_starts",
        "qb_prior_dropbacks": "home_qb_prior_dropbacks",
        "pre_qb_epa_per_dropback": "home_pre_qb_epa_per_dropback",
        "pre_qb_success_rate": "home_pre_qb_success_rate",
        "pre_qb_cpoe": "home_pre_qb_cpoe",
        "pre_qb_sack_rate": "home_pre_qb_sack_rate",
        "pre_qb_interception_rate": "home_pre_qb_interception_rate",
        "pre_qb_explosive_pass_rate": "home_pre_qb_explosive_pass_rate",
    })

    away = q.rename(columns={
        "team": "away_team",
        "qb_id": "away_qb_id",
        "qb_name": "away_qb_name",
        "qb_prior_starts": "away_qb_prior_starts",
        "qb_prior_dropbacks": "away_qb_prior_dropbacks",
        "pre_qb_epa_per_dropback": "away_pre_qb_epa_per_dropback",
        "pre_qb_success_rate": "away_pre_qb_success_rate",
        "pre_qb_cpoe": "away_pre_qb_cpoe",
        "pre_qb_sack_rate": "away_pre_qb_sack_rate",
        "pre_qb_interception_rate": "away_pre_qb_interception_rate",
        "pre_qb_explosive_pass_rate": "away_pre_qb_explosive_pass_rate",
    })

    home_cols = [
        c for c in home.columns
        if c in {"game_id","season","week","home_team"} or c.startswith("home_")
    ]
    away_cols = [
        c for c in away.columns
        if c in {"game_id","season","week","away_team"} or c.startswith("away_")
    ]

    t = t.merge(
        home[home_cols],
        on=["game_id","season","week","home_team"],
        how="left",
        validate="one_to_one",
    )
    t = t.merge(
        away[away_cols],
        on=["game_id","season","week","away_team"],
        how="left",
        validate="one_to_one",
    )

    pairs = [
        ("qb_epa_per_dropback", "diff_qb_epa_per_dropback"),
        ("qb_success_rate", "diff_qb_success_rate"),
        ("qb_cpoe", "diff_qb_cpoe"),
        ("qb_sack_rate", "diff_qb_sack_rate"),
        ("qb_interception_rate", "diff_qb_interception_rate"),
        ("qb_explosive_pass_rate", "diff_qb_explosive_pass_rate"),
    ]
    for base, out_col in pairs:
        t[out_col] = t[f"home_pre_{base}"] - t[f"away_pre_{base}"]

    t["sum_qb_epa_per_dropback"] = (
        t["home_pre_qb_epa_per_dropback"] + t["away_pre_qb_epa_per_dropback"]
    )
    t["sum_qb_success_rate"] = (
        t["home_pre_qb_success_rate"] + t["away_pre_qb_success_rate"]
    )
    t["sum_qb_cpoe"] = t["home_pre_qb_cpoe"] + t["away_pre_qb_cpoe"]
    t["sum_qb_sack_rate"] = (
        t["home_pre_qb_sack_rate"] + t["away_pre_qb_sack_rate"]
    )
    t["sum_qb_interception_rate"] = (
        t["home_pre_qb_interception_rate"] + t["away_pre_qb_interception_rate"]
    )
    t["sum_qb_explosive_pass_rate"] = (
        t["home_pre_qb_explosive_pass_rate"] +
        t["away_pre_qb_explosive_pass_rate"]
    )

    return t


def load_pbp(start_season: int, end_season: int) -> pd.DataFrame:
    frames = []
    for season in range(start_season, end_season + 1):
        url = NFLVERSE_PBP_URL.format(season=season)
        print(f"Loading {season} PBP...")
        frames.append(pd.read_parquet(url))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", default="artifacts/dual_market/training_games_ablation.csv")
    ap.add_argument("--start-season", type=int, default=2018)
    ap.add_argument("--end-season", type=int, default=2025)
    ap.add_argument("--halflife-games", type=float, default=5.0)
    ap.add_argument("--min-games", type=int, default=2)
    ap.add_argument("--output", default="artifacts/dual_market/training_games_qb_v17.csv")
    ap.add_argument("--qb-games-output", default="artifacts/dual_market/qb_games_v17.csv")
    args = ap.parse_args()

    training = pd.read_csv(args.training_csv)

    # Make sure ablation targets / sum features exist even if caller uses the
    # original training file.
    if "actual_margin" not in training:
        training["actual_margin"] = training["home_score"] - training["away_score"]
    if "actual_total" not in training:
        training["actual_total"] = training["home_score"] + training["away_score"]

    pbp = load_pbp(args.start_season, args.end_season)
    qb_games = build_qb_game_table(pbp)
    qb_form = add_pregame_qb_form(
        qb_games,
        halflife_games=args.halflife_games,
        min_games=args.min_games,
    )
    qb_features = build_game_qb_features(qb_form)
    merged = merge_qb_into_training(training, qb_features)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    qbo = Path(args.qb_games_output)
    qbo.parent.mkdir(parents=True, exist_ok=True)
    qb_form.to_csv(qbo, index=False)

    both = merged[
        merged["home_qb_id"].notna() & merged["away_qb_id"].notna()
    ]
    both_form = both[
        both["home_pre_qb_epa_per_dropback"].notna() &
        both["away_pre_qb_epa_per_dropback"].notna()
    ]

    print(f"Saved QB training table: {out}")
    print(f"Rows: {len(merged):,}")
    print(f"Both starters matched: {len(both):,}")
    print(f"Both starters with pregame QB form: {len(both_form):,}")
    print(f"Saved QB game history: {qbo}")


if __name__ == "__main__":
    main()
