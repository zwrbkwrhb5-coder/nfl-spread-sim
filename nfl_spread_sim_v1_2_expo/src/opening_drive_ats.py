from __future__ import annotations
import numpy as np
import pandas as pd

def build_home_opening_drive_ats_study(
    games: pd.DataFrame,
    opening_drives: pd.DataFrame,
    market_closing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Historical research only.

    Measures association between the ACTUAL home opening-drive outcome and
    final margin / ATS result.

    IMPORTANT:
    Actual opening-drive outcome is post-kickoff information and must never be
    used as a pregame prediction feature.
    """
    od = opening_drives.copy()
    home_od = games[["game_id","home_team","home_margin"]].merge(
        od,
        left_on=["game_id","home_team"],
        right_on=["game_id","team"],
        how="left",
    )

    out = home_od.copy()

    if market_closing is not None and not market_closing.empty:
        m = market_closing.copy()
        m = m[m["side_team"].eq(m.get("home_team", m["side_team"]))] if "home_team" in m.columns else m
        # Preferred expected schema: one row per game with home_closing_spread.
        if "home_closing_spread" in m.columns:
            out = out.merge(
                m[["game_id","home_closing_spread"]].drop_duplicates("game_id"),
                on="game_id", how="left"
            )
            out["home_ats_value"] = out["home_margin"] + out["home_closing_spread"]
            out["home_covered"] = np.where(
                out["home_ats_value"] > 0, 1,
                np.where(out["home_ats_value"] < 0, 0, np.nan)
            )

    return out

def summarize_opening_drive_effect(study: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize final margin and ATS by actual first-drive result.
    """
    x = study.copy()
    x["opening_result"] = np.select(
        [
            x["od_td"].eq(1),
            x["od_fg"].eq(1),
            x["od_turnover"].eq(1),
            x["od_scored"].eq(0),
        ],
        [
            "TD",
            "FG",
            "Turnover",
            "No score",
        ],
        default="Other",
    )

    agg = {
        "games": ("game_id","size"),
        "avg_final_home_margin": ("home_margin","mean"),
    }
    if "home_covered" in x.columns:
        agg["ats_cover_rate"] = ("home_covered","mean")

    return (
        x.groupby("opening_result", as_index=False)
        .agg(**agg)
        .sort_values("games", ascending=False)
    )
