from __future__ import annotations
import numpy as np
import pandas as pd

BASE_COLS = [
    "off_epa", "def_epa", "off_success", "def_success",
    "explosive_pass_rate", "sack_allowed_rate", "def_sack_rate",
    "turnover_rate", "points_for", "points_against",
]

ADJ_COLS = [
    "adj_off_epa", "adj_def_epa",
    "adj_off_success", "adj_def_success",
]

def build_team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Create one row per team-game from regular-season PBP.
    Positive off_epa is good offense.
    Lower def_epa is good defense because it is EPA allowed.
    """
    x = pbp.copy()
    x = x[x["season_type"].eq("REG")]
    x = x[x["posteam"].notna() & x["defteam"].notna()]
    x = x[~x["play_type"].isin(["no_play", "kickoff", "extra_point", "field_goal", "punt"])]
    x = x[x["qb_kneel"].fillna(0).eq(0) & x["qb_spike"].fillna(0).eq(0)]

    x["is_pass"] = x["pass"].fillna(0).eq(1)
    x["explosive_pass"] = x["is_pass"] & x["yards_gained"].fillna(0).ge(20)
    x["turnover"] = (
        x["interception"].fillna(0).eq(1) |
        x["fumble_lost"].fillna(0).eq(1)
    )

    off = (
        x.groupby(["game_id", "season", "week", "posteam"], as_index=False)
        .agg(
            off_epa=("epa", "mean"),
            off_success=("success", "mean"),
            explosive_pass_rate=("explosive_pass", "mean"),
            sack_allowed_rate=("sack", "mean"),
            turnover_rate=("turnover", "mean"),
            plays=("epa", "size"),
        )
        .rename(columns={"posteam": "team"})
    )

    deff = (
        x.groupby(["game_id", "season", "week", "defteam"], as_index=False)
        .agg(
            def_epa=("epa", "mean"),
            def_success=("success", "mean"),
            def_sack_rate=("sack", "mean"),
        )
        .rename(columns={"defteam": "team"})
    )

    scores = (
        pbp[pbp["season_type"].eq("REG")]
        .groupby(["game_id", "season", "week", "home_team", "away_team"], as_index=False)
        .agg(home_score=("home_score", "max"), away_score=("away_score", "max"))
    )

    home = scores[["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]].copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["is_home"] = 1
    home["points_for"] = home["home_score"]
    home["points_against"] = home["away_score"]

    away = scores[["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]].copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["is_home"] = 0
    away["points_for"] = away["away_score"]
    away["points_against"] = away["home_score"]

    keep = [
        "game_id", "season", "week", "team", "opponent", "is_home",
        "home_score", "away_score", "points_for", "points_against"
    ]
    tg = pd.concat([home[keep], away[keep]], ignore_index=True)
    tg = tg.merge(off, on=["game_id", "season", "week", "team"], how="left")
    tg = tg.merge(deff, on=["game_id", "season", "week", "team"], how="left")
    tg = tg.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    return tg


def _ewm_shifted(series: pd.Series, halflife_games: float, min_games: int) -> pd.Series:
    """
    Exponentially weighted pregame mean.
    shift(1) guarantees the current game's value is never used.
    """
    return (
        series.shift(1)
        .ewm(halflife=halflife_games, adjust=False, min_periods=min_games)
        .mean()
    )


def add_pregame_recency_features(
    team_games: pd.DataFrame,
    halflife_games: float = 6.0,
    min_games: int = 3,
) -> pd.DataFrame:
    """
    Add exponentially decayed pregame features.

    A half-life of 6 games means a game 6 team-games ago has half the weight
    of the newest completed game, before normalization.
    """
    df = team_games.sort_values(["team", "season", "week", "game_id"]).copy()

    for col in BASE_COLS:
        df[f"pre_{col}"] = (
            df.groupby("team", group_keys=False)[col]
            .apply(lambda s: _ewm_shifted(s, halflife_games, min_games))
        )

    return df


def add_opponent_adjusted_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Opponent-adjust each completed game's EPA and success-rate performance using
    the opponent's PRE-game strength.

    Example:
      offense produces +0.10 EPA/play vs a defense that entered at -0.08 EPA/play allowed.
      adjusted offense = +0.10 - (-0.08) = +0.18

    For defense, lower is better:
      defense allows 0.00 EPA/play vs offense entering at +0.12
      adjusted defense = 0.00 - 0.12 = -0.12
    """
    df = team_games.copy()

    opp_cols = {
        "pre_off_epa": "opp_pre_off_epa",
        "pre_def_epa": "opp_pre_def_epa",
        "pre_off_success": "opp_pre_off_success",
        "pre_def_success": "opp_pre_def_success",
    }

    opp = df[["game_id", "team"] + list(opp_cols.keys())].copy()
    opp = opp.rename(columns={"team": "opponent", **opp_cols})

    df = df.merge(opp, on=["game_id", "opponent"], how="left")

    df["adj_off_epa"] = df["off_epa"] - df["opp_pre_def_epa"]
    df["adj_def_epa"] = df["def_epa"] - df["opp_pre_off_epa"]

    df["adj_off_success"] = df["off_success"] - df["opp_pre_def_success"]
    df["adj_def_success"] = df["def_success"] - df["opp_pre_off_success"]

    return df


def add_pregame_adjusted_recency_features(
    team_games: pd.DataFrame,
    halflife_games: float = 6.0,
    min_games: int = 3,
) -> pd.DataFrame:
    """
    Add pregame EWM averages of opponent-adjusted game performances.
    """
    df = team_games.sort_values(["team", "season", "week", "game_id"]).copy()

    for col in ADJ_COLS:
        df[f"pre_{col}"] = (
            df.groupby("team", group_keys=False)[col]
            .apply(lambda s: _ewm_shifted(s, halflife_games, min_games))
        )
    return df


def build_features(
    team_games: pd.DataFrame,
    halflife_games: float = 6.0,
    min_games: int = 3,
) -> pd.DataFrame:
    """
    Full v0.2 feature pipeline:
      1) raw exponentially weighted pregame strength
      2) opponent-adjust each completed game
      3) exponentially weight the opponent-adjusted performances
    """
    df = add_pregame_recency_features(
        team_games, halflife_games=halflife_games, min_games=min_games
    )
    df = add_opponent_adjusted_features(df)
    df = add_pregame_adjusted_recency_features(
        df, halflife_games=halflife_games, min_games=min_games
    )
    return df


def build_matchups(team_games: pd.DataFrame):
    """
    Convert two team rows into one game row using HOME minus AWAY features.
    """
    df = team_games.copy()
    feature_bases = BASE_COLS + ADJ_COLS
    pre_cols = [f"pre_{c}" for c in feature_bases]

    homes = df[df["is_home"].eq(1)].copy()
    aways = df[df["is_home"].eq(0)].copy()

    h = homes[
        ["game_id", "season", "week", "team", "opponent",
         "home_score", "away_score"] + pre_cols
    ].copy()
    h = h.rename(
        columns={
            "team": "home_team",
            "opponent": "away_team",
            **{c: f"home_{c}" for c in pre_cols},
        }
    )

    a = aways[["game_id", "team"] + pre_cols].copy()
    a = a.rename(
        columns={
            "team": "away_team_check",
            **{c: f"away_{c}" for c in pre_cols},
        }
    )

    g = h.merge(a, on="game_id", how="inner")
    g = g[g["away_team"].eq(g["away_team_check"])].copy()

    model_features = []
    for base in feature_bases:
        c = f"pre_{base}"
        out = f"diff_{base}"
        g[out] = g[f"home_{c}"] - g[f"away_{c}"]
        model_features.append(out)

    g["home_margin"] = g["home_score"] - g["away_score"]

    g = g.dropna(subset=model_features + ["home_margin"]).reset_index(drop=True)
    return g, model_features


def merge_qb_features_into_matchups(
    games: pd.DataFrame,
    qb_games: pd.DataFrame,
):
    """
    Merge starting-QB pregame form into one-row-per-game matchup data.

    Uses starter approximation = QB with most dropbacks in that team-game.
    For each game, HOME QB minus AWAY QB features are created.

    Returns:
      merged_games, qb_feature_names
    """
    q = qb_games[qb_games["starter"].eq(1)].copy()

    qb_pre_cols = [
        "pre_qb_epa_per_dropback",
        "pre_qb_success_rate",
        "pre_qb_cpoe",
        "pre_qb_sack_rate",
        "pre_qb_explosive_pass_rate",
        "pre_qb_int_rate",
    ]

    available = [c for c in qb_pre_cols if c in q.columns]

    home_qb = q[["game_id", "team", "player_id", "player_name"] + available].copy()
    home_qb = home_qb.rename(
        columns={
            "team": "home_team",
            "player_id": "home_qb_id",
            "player_name": "home_qb_name",
            **{c: f"home_{c}" for c in available},
        }
    )

    away_qb = q[["game_id", "team", "player_id", "player_name"] + available].copy()
    away_qb = away_qb.rename(
        columns={
            "team": "away_team",
            "player_id": "away_qb_id",
            "player_name": "away_qb_name",
            **{c: f"away_{c}" for c in available},
        }
    )

    out = games.merge(home_qb, on=["game_id", "home_team"], how="left")
    out = out.merge(away_qb, on=["game_id", "away_team"], how="left")

    qb_features = []
    for c in available:
        base = c.removeprefix("pre_")
        name = f"diff_{base}"
        out[name] = out[f"home_{c}"] - out[f"away_{c}"]
        qb_features.append(name)

    return out, qb_features
