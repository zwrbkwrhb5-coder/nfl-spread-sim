from __future__ import annotations
import numpy as np
import pandas as pd

INJURY_COLUMNS = [
    "player_id",
    "player_name",
    "season",
    "injury_start_week",
    "return_week",
    "injury_group",
    "games_missed",
]

def validate_injury_events(injuries: pd.DataFrame) -> pd.DataFrame:
    """
    Expected one row per QB injury absence episode.

    injury_group examples:
      ankle, hamstring, knee, shoulder, throwing_arm, hand_finger,
      concussion, rib_torso, foot, illness_other
    """
    missing = [c for c in INJURY_COLUMNS if c not in injuries.columns]
    if missing:
        raise ValueError(f"Missing injury columns: {missing}")

    out = injuries.copy()
    out["games_missed"] = pd.to_numeric(out["games_missed"], errors="coerce")
    return out


def attach_return_events(
    qb_games: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Marks a QB game as the first game back when season/week match an injury event.

    IMPORTANT:
    Return-game outcome metrics are for historical research only.
    They must never be used as inputs to predict that same return game.
    """
    inj = validate_injury_events(injuries)

    keys = [
        "player_id", "season", "return_week",
        "injury_group", "games_missed"
    ]
    ev = inj[keys].rename(columns={"return_week": "week"}).copy()
    ev["first_game_back"] = 1

    q = qb_games.merge(
        ev,
        on=["player_id", "season", "week"],
        how="left",
    )

    q["first_game_back"] = q["first_game_back"].fillna(0).astype(int)
    q["games_missed"] = q["games_missed"].fillna(0)
    q["injury_group"] = q["injury_group"].fillna("none")
    return q


def build_return_performance_table(
    qb_games_with_returns: pd.DataFrame,
    preinjury_lookback: int = 4,
) -> pd.DataFrame:
    """
    Historical research table measuring first-game-back performance vs the QB's
    recent pre-injury baseline.

    This table is used to LEARN priors for future return games.
    """
    q = qb_games_with_returns.sort_values(
        ["player_id", "season", "week", "game_id"]
    ).copy()

    rows = []
    for idx, r in q[q["first_game_back"].eq(1)].iterrows():
        hist = q[
            (q["player_id"].eq(r["player_id"])) &
            (
                (q["season"] < r["season"]) |
                ((q["season"] == r["season"]) & (q["week"] < r["week"]))
            )
        ].tail(preinjury_lookback)

        if hist.empty:
            continue

        pre_epa = hist["qb_epa_per_dropback"].mean()
        pre_success = hist["qb_success_rate"].mean()
        pre_cpoe = hist["qb_cpoe"].mean()
        pre_sack = hist["qb_sack_rate"].mean()

        rows.append({
            "player_id": r["player_id"],
            "player_name": r.get("player_name"),
            "season": r["season"],
            "return_week": r["week"],
            "injury_group": r["injury_group"],
            "games_missed": r["games_missed"],
            "preinjury_epa_db": pre_epa,
            "return_epa_db": r["qb_epa_per_dropback"],
            "epa_db_delta": r["qb_epa_per_dropback"] - pre_epa,
            "preinjury_success": pre_success,
            "return_success": r["qb_success_rate"],
            "success_delta": r["qb_success_rate"] - pre_success,
            "preinjury_cpoe": pre_cpoe,
            "return_cpoe": r["qb_cpoe"],
            "cpoe_delta": r["qb_cpoe"] - pre_cpoe,
            "preinjury_sack_rate": pre_sack,
            "return_sack_rate": r["qb_sack_rate"],
            "sack_rate_delta": r["qb_sack_rate"] - pre_sack,
        })

    return pd.DataFrame(rows)


def _missed_bucket(games_missed: float) -> str:
    if games_missed <= 1:
        return "1"
    if games_missed <= 3:
        return "2_3"
    if games_missed <= 6:
        return "4_6"
    return "7_plus"


def build_return_priors(return_table: pd.DataFrame) -> pd.DataFrame:
    """
    Cohort priors by injury type and games-missed bucket.

    These priors are designed to answer:
      'Historically, what happens in the first game back after this kind of
       injury and this much time missed?'
    """
    if return_table.empty:
        return pd.DataFrame()

    r = return_table.copy()
    r["missed_bucket"] = r["games_missed"].map(_missed_bucket)

    priors = (
        r.groupby(["injury_group", "missed_bucket"], as_index=False)
        .agg(
            sample_size=("player_id", "size"),
            mean_epa_db_delta=("epa_db_delta", "mean"),
            median_epa_db_delta=("epa_db_delta", "median"),
            mean_success_delta=("success_delta", "mean"),
            mean_cpoe_delta=("cpoe_delta", "mean"),
            mean_sack_rate_delta=("sack_rate_delta", "mean"),
            epa_delta_std=("epa_db_delta", "std"),
        )
    )
    return priors


def get_return_adjustment(
    priors: pd.DataFrame,
    injury_group: str,
    games_missed: int,
    min_sample: int = 8,
    shrinkage_games: int = 20,
) -> dict:
    """
    Get a conservative first-game-back QB adjustment.

    Small samples are shrunk toward zero so the model does not overreact.
    """
    bucket = _missed_bucket(games_missed)
    row = priors[
        priors["injury_group"].eq(injury_group) &
        priors["missed_bucket"].eq(bucket)
    ]

    if row.empty:
        return {
            "qb_return_epa_adjustment": 0.0,
            "qb_return_success_adjustment": 0.0,
            "qb_return_cpoe_adjustment": 0.0,
            "qb_return_sack_adjustment": 0.0,
            "return_prior_sample": 0,
            "return_prior_reliability": 0.0,
        }

    r = row.iloc[0]
    n = int(r["sample_size"])
    reliability = n / (n + shrinkage_games)

    # If extremely small sample, shrink harder.
    if n < min_sample:
        reliability *= 0.5

    return {
        "qb_return_epa_adjustment":
            float(r["mean_epa_db_delta"] * reliability),
        "qb_return_success_adjustment":
            float(r["mean_success_delta"] * reliability),
        "qb_return_cpoe_adjustment":
            float(r["mean_cpoe_delta"] * reliability),
        "qb_return_sack_adjustment":
            float(r["mean_sack_rate_delta"] * reliability),
        "return_prior_sample": n,
        "return_prior_reliability": float(reliability),
    }
