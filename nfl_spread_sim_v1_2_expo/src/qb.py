from __future__ import annotations
import numpy as np
import pandas as pd

QB_METRICS = [
    "qb_epa_per_dropback",
    "qb_success_rate",
    "qb_cpoe",
    "qb_sack_rate",
    "qb_explosive_pass_rate",
    "qb_int_rate",
]

def build_qb_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per QB-team-game.

    Starter is approximated as the QB with the most dropbacks for that team-game.
    This is intentionally transparent and can later be replaced with an official
    starter/depth-chart source.
    """
    x = pbp.copy()
    x = x[x.get("season_type", "REG").eq("REG")]
    x = x[x["posteam"].notna()]

    if "passer_player_id" not in x.columns:
        raise ValueError("PBP does not contain passer_player_id.")

    x = x[x["passer_player_id"].notna()].copy()
    x["dropback"] = (
        x.get("pass", 0).fillna(0).eq(1) |
        x.get("sack", 0).fillna(0).eq(1)
    )
    x = x[x["dropback"]].copy()

    x["explosive_pass"] = (
        x.get("pass", 0).fillna(0).eq(1) &
        x.get("yards_gained", 0).fillna(0).ge(20)
    )

    agg_map = {
        "dropbacks": ("dropback", "sum"),
        "qb_epa_per_dropback": ("epa", "mean"),
        "qb_success_rate": ("success", "mean"),
        "qb_sack_rate": ("sack", "mean"),
        "qb_explosive_pass_rate": ("explosive_pass", "mean"),
        "qb_int_rate": ("interception", "mean"),
    }
    if "cpoe" in x.columns:
        agg_map["qb_cpoe"] = ("cpoe", "mean")

    q = (
        x.groupby(
            ["game_id", "season", "week", "posteam",
             "passer_player_id", "passer_player_name"],
            as_index=False,
        )
        .agg(**agg_map)
        .rename(columns={
            "posteam": "team",
            "passer_player_id": "player_id",
            "passer_player_name": "player_name",
        })
    )

    if "qb_cpoe" not in q.columns:
        q["qb_cpoe"] = np.nan

    q["starter"] = (
        q.groupby(["game_id", "team"])["dropbacks"]
        .transform("max")
        .eq(q["dropbacks"])
    ).astype(int)

    return q.sort_values(["player_id", "season", "week", "game_id"]).reset_index(drop=True)


def add_qb_pregame_form(
    qb_games: pd.DataFrame,
    halflife_games: float = 5.0,
    min_games: int = 2,
) -> pd.DataFrame:
    """
    Add shifted exponentially weighted QB form.
    Current-game performance is never used in its own prediction row.
    """
    df = qb_games.copy().sort_values(["player_id", "season", "week", "game_id"])

    for col in QB_METRICS:
        df[f"pre_{col}"] = (
            df.groupby("player_id", group_keys=False)[col]
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


def build_current_qb_state(
    qb_games: pd.DataFrame,
    player_id: str,
    halflife_games: float = 5.0,
    min_games: int = 2,
) -> dict:
    """
    State for a FUTURE game. All completed games are legitimately available.
    """
    rows = qb_games[qb_games["player_id"].eq(player_id)].sort_values(
        ["season", "week", "game_id"]
    )
    if len(rows) < min_games:
        raise ValueError(
            f"Need at least {min_games} QB games for {player_id}; found {len(rows)}"
        )

    state = {}
    for col in QB_METRICS:
        s = rows[col].dropna()
        if len(s) < min_games:
            state[col] = np.nan
        else:
            state[col] = float(
                s.ewm(
                    halflife=halflife_games,
                    adjust=False,
                    min_periods=min_games,
                ).mean().iloc[-1]
            )
    return state
