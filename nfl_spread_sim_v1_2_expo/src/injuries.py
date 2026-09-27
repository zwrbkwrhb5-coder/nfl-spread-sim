from __future__ import annotations
import numpy as np
import pandas as pd

POSITION_GROUPS = {
    "QB": "qb",
    "RB": "rb",
    "FB": "rb",
    "WR": "wr_te",
    "TE": "wr_te",
    "LT": "ol",
    "LG": "ol",
    "C": "ol",
    "RG": "ol",
    "RT": "ol",
    "OL": "ol",
    "DE": "pass_rush",
    "EDGE": "pass_rush",
    "OLB": "pass_rush",
    "DT": "interior_dl",
    "NT": "interior_dl",
    "ILB": "lb",
    "LB": "lb",
    "CB": "secondary",
    "S": "secondary",
    "FS": "secondary",
    "SS": "secondary",
    "DB": "secondary",
    "K": "special_teams",
    "P": "special_teams",
}

DEFAULT_POSITION_IMPORTANCE = {
    "qb": 1.00,
    "ol": 0.40,
    "wr_te": 0.35,
    "pass_rush": 0.35,
    "secondary": 0.30,
    "interior_dl": 0.20,
    "lb": 0.18,
    "rb": 0.12,
    "special_teams": 0.08,
    "other": 0.10,
}

STATUS_SEVERITY = {
    "out": 1.00,
    "ir": 1.00,
    "inactive": 1.00,
    "doubtful": 0.75,
    "questionable": 0.35,
    "limited": 0.20,
    "probable": 0.10,
    "full": 0.00,
    "active": 0.00,
}

def normalize_position_group(position: str) -> str:
    if position is None or (isinstance(position, float) and np.isnan(position)):
        return "other"
    return POSITION_GROUPS.get(str(position).upper().strip(), "other")

def normalize_status(status: str) -> str:
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return "active"
    return str(status).lower().strip()

def validate_player_availability(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "game_id", "season", "week", "team", "player_id",
        "player_name", "position", "status",
        "snap_share_prior", "starter_flag", "player_value_prior",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing availability columns: {missing}")

    x = df.copy()
    x["position_group"] = x["position"].map(normalize_position_group)
    x["status_norm"] = x["status"].map(normalize_status)
    x["snap_share_prior"] = pd.to_numeric(x["snap_share_prior"], errors="coerce").fillna(0.0)
    x["starter_flag"] = pd.to_numeric(x["starter_flag"], errors="coerce").fillna(0.0)
    x["player_value_prior"] = pd.to_numeric(x["player_value_prior"], errors="coerce").fillna(0.0)
    x["status_severity"] = x["status_norm"].map(STATUS_SEVERITY).fillna(0.0)
    return x

def player_impact_score(
    row: pd.Series,
    position_importance: dict | None = None,
) -> float:
    """
    Experimental importance score.

    Inputs:
    - position importance prior
    - prior snap share
    - starter flag
    - learned/estimated player value prior
    - injury/inactive severity

    This is not a point-spread adjustment. It is an input feature.
    The model must learn the relationship from historical games.
    """
    imp = position_importance or DEFAULT_POSITION_IMPORTANCE
    group = row["position_group"]
    base = float(imp.get(group, imp["other"]))
    snap = float(np.clip(row["snap_share_prior"], 0, 1))
    starter = float(np.clip(row["starter_flag"], 0, 1))
    val = float(max(row["player_value_prior"], 0))
    severity = float(np.clip(row["status_severity"], 0, 1))

    role_weight = 0.55 * snap + 0.25 * starter + 0.20 * val
    return base * role_weight * severity

def build_team_game_injury_features(
    availability: pd.DataFrame,
    position_importance: dict | None = None,
) -> pd.DataFrame:
    """
    Aggregate player-level availability into team-game injury features.
    """
    x = validate_player_availability(availability)
    x["impact_score"] = x.apply(
        lambda r: player_impact_score(r, position_importance),
        axis=1,
    )

    x["is_out_like"] = x["status_severity"].ge(0.75).astype(int)
    x["is_questionable_like"] = (
        x["status_severity"].gt(0) & x["status_severity"].lt(0.75)
    ).astype(int)

    agg = (
        x.groupby(["game_id", "season", "week", "team"], as_index=False)
        .agg(
            injury_total_impact=("impact_score", "sum"),
            injury_out_count=("is_out_like", "sum"),
            injury_questionable_count=("is_questionable_like", "sum"),
        )
    )

    # Position-group impact features.
    piv = (
        x.pivot_table(
            index=["game_id", "season", "week", "team"],
            columns="position_group",
            values="impact_score",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    piv.columns = [
        c if isinstance(c, str) else c
        for c in piv.columns
    ]

    for col in list(piv.columns):
        if col not in ["game_id", "season", "week", "team"]:
            piv = piv.rename(columns={col: f"injury_{col}_impact"})

    out = agg.merge(
        piv, on=["game_id", "season", "week", "team"], how="left"
    )

    return out

def merge_injury_features_into_games(
    games: pd.DataFrame,
    team_injuries: pd.DataFrame,
):
    """
    Merge home/away injury features and create home-minus-away differentials.
    """
    g = games.copy()

    base_cols = [
        c for c in team_injuries.columns
        if c not in ["game_id", "season", "week", "team"]
    ]

    home = team_injuries.rename(
        columns={
            "team": "home_team",
            **{c: f"home_{c}" for c in base_cols},
        }
    )
    away = team_injuries.rename(
        columns={
            "team": "away_team",
            **{c: f"away_{c}" for c in base_cols},
        }
    )

    g = g.merge(
        home,
        on=["game_id", "season", "week", "home_team"],
        how="left",
    )
    g = g.merge(
        away,
        on=["game_id", "season", "week", "away_team"],
        how="left",
    )

    injury_features = []
    for c in base_cols:
        hc = f"home_{c}"
        ac = f"away_{c}"
        if hc not in g.columns or ac not in g.columns:
            continue
        g[hc] = g[hc].fillna(0.0)
        g[ac] = g[ac].fillna(0.0)
        name = f"diff_{c}"
        # Positive means HOME has more injury burden than AWAY.
        g[name] = g[hc] - g[ac]
        injury_features.append(name)

    return g, injury_features
