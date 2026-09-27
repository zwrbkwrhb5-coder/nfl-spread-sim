from __future__ import annotations
import numpy as np
import pandas as pd

def american_to_break_even(odds: int | float) -> float:
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)

def american_to_decimal(odds: int | float) -> float:
    odds = float(odds)
    if odds < 0:
        return 1.0 + 100.0 / (-odds)
    return 1.0 + odds / 100.0

def validate_market_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected one row per sportsbook/side/snapshot.

    Required:
      game_id
      snapshot_time
      sportsbook
      side_team
      spread
      american_odds
      market_phase  # opening/current/closing or timestamp label
    """
    required = [
        "game_id", "snapshot_time", "sportsbook",
        "side_team", "spread", "american_odds", "market_phase"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing market columns: {missing}")

    x = df.copy()
    x["snapshot_time"] = pd.to_datetime(x["snapshot_time"], errors="coerce", utc=True)
    x["spread"] = pd.to_numeric(x["spread"], errors="coerce")
    x["american_odds"] = pd.to_numeric(x["american_odds"], errors="coerce")
    x["break_even_probability"] = x["american_odds"].map(american_to_break_even)
    x["decimal_odds"] = x["american_odds"].map(american_to_decimal)
    return x

def build_consensus_line(
    market: pd.DataFrame,
    phase: str = "current",
) -> pd.DataFrame:
    """
    Median spread and median price across books for each game/side.
    """
    x = validate_market_snapshot(market)
    x = x[x["market_phase"].astype(str).str.lower().eq(phase.lower())].copy()

    if x.empty:
        return pd.DataFrame()

    return (
        x.groupby(["game_id", "side_team"], as_index=False)
        .agg(
            consensus_spread=("spread", "median"),
            consensus_american_odds=("american_odds", "median"),
            books=("sportsbook", "nunique"),
            latest_snapshot=("snapshot_time", "max"),
        )
    )

def build_line_movement(market: pd.DataFrame) -> pd.DataFrame:
    """
    Compare opening vs current vs closing consensus spread.
    """
    x = validate_market_snapshot(market)

    rows = []
    for (game_id, side), g in x.groupby(["game_id", "side_team"]):
        def median_for(phase):
            z = g[g["market_phase"].astype(str).str.lower().eq(phase)]
            if z.empty:
                return np.nan
            return float(z["spread"].median())

        opening = median_for("opening")
        current = median_for("current")
        closing = median_for("closing")

        rows.append({
            "game_id": game_id,
            "side_team": side,
            "opening_spread": opening,
            "current_spread": current,
            "closing_spread": closing,
            "move_open_to_current": (
                current - opening if np.isfinite(current) and np.isfinite(opening) else np.nan
            ),
            "move_open_to_close": (
                closing - opening if np.isfinite(closing) and np.isfinite(opening) else np.nan
            ),
        })
    return pd.DataFrame(rows)

def best_available_line(
    market: pd.DataFrame,
    game_id: str,
    side_team: str,
    phase: str = "current",
) -> pd.DataFrame:
    """
    Rank books for a side.

    Priority:
    1. Most favorable spread for that side.
       Example favorite: -2.5 better than -3.5
       Example underdog: +4.5 better than +3.5
       In both cases, numerically larger spread is better for the bettor.
    2. Better price (higher American odds / lower implied break-even).
    """
    x = validate_market_snapshot(market)
    x = x[
        x["game_id"].eq(game_id) &
        x["side_team"].eq(side_team) &
        x["market_phase"].astype(str).str.lower().eq(phase.lower())
    ].copy()

    if x.empty:
        return x

    return x.sort_values(
        ["spread", "break_even_probability"],
        ascending=[False, True]
    ).reset_index(drop=True)

def add_market_model_edges(
    market: pd.DataFrame,
    projected_margin_by_side: dict,
    simulated_cover_probability_fn,
    phase: str = "current",
) -> pd.DataFrame:
    """
    For every available book/side, calculate:
      - projected differential
      - simulated cover probability
      - implied break-even
      - probability edge

    projected_margin_by_side:
      {team: projected margin from that team's perspective}
      Example BUF expected to win by 6.2 => {"BUF": 6.2, "MIA": -6.2}

    simulated_cover_probability_fn(side_team, spread) -> probability
    """
    x = validate_market_snapshot(market)
    x = x[x["market_phase"].astype(str).str.lower().eq(phase.lower())].copy()

    rows = []
    for _, r in x.iterrows():
        side = r["side_team"]
        if side not in projected_margin_by_side:
            continue
        model_margin = float(projected_margin_by_side[side])
        spread = float(r["spread"])
        cover_prob = float(simulated_cover_probability_fn(side, spread))
        breakeven = float(r["break_even_probability"])

        rec = r.to_dict()
        rec.update({
            "model_margin_for_side": model_margin,
            # Example: model says +6.2 and market spread is -3.5 => 2.7 pts model cushion.
            "projected_differential": model_margin + spread,
            "sim_cover_probability": cover_prob,
            "probability_edge": cover_prob - breakeven,
        })
        rows.append(rec)

    return pd.DataFrame(rows)

def rank_top_edges(edges: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Rank by calibrated/simulated probability edge first, then projected differential.
    One row per game/side/book opportunity.
    """
    if edges.empty:
        return edges
    x = edges.copy()
    cols = ["probability_edge", "projected_differential"]
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.sort_values(
        ["probability_edge", "projected_differential"],
        ascending=[False, False],
    ).head(top_n).reset_index(drop=True)
