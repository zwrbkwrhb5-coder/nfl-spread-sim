from __future__ import annotations
import numpy as np
import pandas as pd

OD_METRICS = [
    "od_points",
    "od_scored",
    "od_td",
    "od_fg",
    "od_turnover",
    "od_epa",
]

def identify_opening_drives(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per team-game for the team's FIRST offensive drive.

    Uses the first observed possession for each team/game. The result is
    historical outcome data and must only be used to create shifted pregame
    tendencies, never directly as a current-game feature.
    """
    x = pbp.copy()
    x = x[x["season_type"].eq("REG")]
    x = x[x["posteam"].notna()].copy()

    # If drive number exists, use it. Otherwise infer by earliest play index.
    if "drive" in x.columns:
        x["drive_num"] = pd.to_numeric(x["drive"], errors="coerce")
    else:
        x["drive_num"] = np.nan

    # Sort stably by original row order if no clock/order column is guaranteed.
    x = x.reset_index().rename(columns={"index":"play_order"})
    x = x.sort_values(["game_id","posteam","play_order"])

    first_drive_id = (
        x.groupby(["game_id","posteam"])["drive_num"]
        .transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    )

    if first_drive_id.notna().any():
        first = x[
            (x["drive_num"].eq(first_drive_id)) |
            (first_drive_id.isna() & x.groupby(["game_id","posteam"]).cumcount().lt(20))
        ].copy()
    else:
        # Conservative fallback: first 20 offensive plays is only a schema fallback.
        first = x[x.groupby(["game_id","posteam"]).cumcount().lt(20)].copy()

    first["turnover_flag"] = (
        first.get("interception", 0).fillna(0).eq(1) |
        first.get("fumble_lost", 0).fillna(0).eq(1)
    )

    # Infer scoring from score changes during the drive when posteam score exists.
    def agg_drive(g):
        start_score = g["posteam_score"].dropna().iloc[0] if g["posteam_score"].notna().any() else np.nan
        end_score = g["posteam_score"].dropna().iloc[-1] if g["posteam_score"].notna().any() else np.nan
        pts = float(end_score - start_score) if np.isfinite(start_score) and np.isfinite(end_score) else 0.0
        return pd.Series({
            "od_points": pts,
            "od_scored": float(pts > 0),
            "od_td": float(pts >= 6),
            "od_fg": float(2 < pts < 6),
            "od_turnover": float(g["turnover_flag"].any()),
            "od_epa": float(g["epa"].mean()) if g["epa"].notna().any() else np.nan,
            "od_plays": int(len(g)),
        })

    out = (
        first.groupby(["game_id","season","week","posteam"], as_index=False)
        .apply(agg_drive, include_groups=False)
        .reset_index()
    )

    if "level_4" in out.columns:
        out = out.drop(columns=["level_4"])
    out = out.rename(columns={"posteam":"team"})
    return out

def add_opening_drive_pregame_form(
    opening_drives: pd.DataFrame,
    halflife_games: float = 6.0,
    min_games: int = 3,
) -> pd.DataFrame:
    """
    Shifted exponentially weighted opening-drive tendencies.
    """
    df = opening_drives.sort_values(["team","season","week","game_id"]).copy()
    for col in OD_METRICS:
        df[f"pre_{col}"] = (
            df.groupby("team", group_keys=False)[col]
              .apply(
                  lambda s: s.shift(1)
                  .ewm(
                      halflife=halflife_games,
                      adjust=False,
                      min_periods=min_games,
                  )
                  .mean()
              )
        )
    return df

def merge_opening_drive_features(
    games: pd.DataFrame,
    opening_drive_form: pd.DataFrame,
):
    """
    Merge home and away pregame OD tendencies into game rows.

    Also creates interaction features between one team's opening-drive offense
    and the opponent's opening-drive prevention history when available.

    Because the same OD table contains only offensive outcomes, defensive
    prevention is approximated by the opponent's opponents' first-drive results
    in a separate helper.
    """
    q = opening_drive_form.copy()
    pre_cols = [f"pre_{c}" for c in OD_METRICS if f"pre_{c}" in q.columns]

    home = q[["game_id","team"] + pre_cols].rename(
        columns={"team":"home_team", **{c:f"home_{c}" for c in pre_cols}}
    )
    away = q[["game_id","team"] + pre_cols].rename(
        columns={"team":"away_team", **{c:f"away_{c}" for c in pre_cols}}
    )

    g = games.merge(home, on=["game_id","home_team"], how="left")
    g = g.merge(away, on=["game_id","away_team"], how="left")

    features = []
    for c in pre_cols:
        base = c.removeprefix("pre_")
        name = f"diff_od_{base}"
        g[name] = g[f"home_{c}"] - g[f"away_{c}"]
        features.append(name)

    return g, features

def build_opening_drive_defense_form(
    opening_drives: pd.DataFrame,
    game_teams: pd.DataFrame,
    halflife_games: float = 6.0,
    min_games: int = 3,
) -> pd.DataFrame:
    """
    Convert offensive opening-drive outcomes into defensive opening-drive
    allowed tendencies for the opposing team.

    game_teams expected:
      game_id, home_team, away_team
    """
    x = opening_drives.merge(game_teams, on="game_id", how="left")

    def opponent(row):
        if row["team"] == row["home_team"]:
            return row["away_team"]
        if row["team"] == row["away_team"]:
            return row["home_team"]
        return np.nan

    x["def_team"] = x.apply(opponent, axis=1)
    x = x[x["def_team"].notna()].copy()

    ren = {
        "od_points":"od_points_allowed",
        "od_scored":"od_score_allowed",
        "od_td":"od_td_allowed",
        "od_fg":"od_fg_allowed",
        "od_turnover":"od_turnover_forced",
        "od_epa":"od_epa_allowed",
    }
    d = x[["game_id","season","week","def_team"] + list(ren.keys())].rename(
        columns={"def_team":"team", **ren}
    )

    for col in ren.values():
        d[f"pre_{col}"] = (
            d.sort_values(["team","season","week","game_id"])
             .groupby("team", group_keys=False)[col]
             .apply(
                 lambda s: s.shift(1)
                 .ewm(
                     halflife=halflife_games,
                     adjust=False,
                     min_periods=min_games,
                 )
                 .mean()
             )
        )
    return d

def merge_opening_drive_defense(
    games: pd.DataFrame,
    defense_form: pd.DataFrame,
):
    pre_cols = [c for c in defense_form.columns if c.startswith("pre_od_")]

    home = defense_form[["game_id","team"] + pre_cols].rename(
        columns={"team":"home_team", **{c:f"home_{c}" for c in pre_cols}}
    )
    away = defense_form[["game_id","team"] + pre_cols].rename(
        columns={"team":"away_team", **{c:f"away_{c}" for c in pre_cols}}
    )

    g = games.merge(home, on=["game_id","home_team"], how="left")
    g = g.merge(away, on=["game_id","away_team"], how="left")

    feats = []
    for c in pre_cols:
        base = c.removeprefix("pre_")
        name = f"diff_{base}"
        g[name] = g[f"home_{c}"] - g[f"away_{c}"]
        feats.append(name)

    # Direct matchup signals
    if "home_pre_od_scored" in g.columns and "away_pre_od_score_allowed" in g.columns:
        g["home_opening_score_matchup"] = (
            g["home_pre_od_scored"] + g["away_pre_od_score_allowed"]
        ) / 2.0
        feats.append("home_opening_score_matchup")

    if "away_pre_od_scored" in g.columns and "home_pre_od_score_allowed" in g.columns:
        g["away_opening_score_matchup"] = (
            g["away_pre_od_scored"] + g["home_pre_od_score_allowed"]
        ) / 2.0
        feats.append("away_opening_score_matchup")

    return g, feats
