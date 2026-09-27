from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_LEDGER = "artifacts/live/forward_test_ledger_v39.csv"


LEDGER_TEXT_COLUMNS = [
    "game_id",
    "away_team",
    "home_team",
    "market",
    "pick",
    "book",
    "quote_updated_at",
    "away_qb_name",
    "home_qb_name",
    "entry_snapshot_label",
    "captured_utc",
    "model_version",
    "record_type",
    "status",
    "closing_snapshot_label",
    "closing_captured_utc",
    "close_best_book",
    "result",
]


def normalize_ledger_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    CSV round-trips turn all-empty text columns into float64.
    Pandas 3.x rejects later string assignment into those float columns.

    Restore known ledger text fields to object dtype after every read.
    """
    out = df.copy()

    for c in LEDGER_TEXT_COLUMNS:
        if c in out.columns:
            out[c] = out[c].astype("object")

    return out


def american_break_even(odds):
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def american_profit_per_unit(odds):
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    return 100.0 / (-odds)


def truthy(s):
    if pd.isna(s):
        return False
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in {"1", "true", "yes", "y"}


def append_unique(existing: pd.DataFrame, new: pd.DataFrame, keys):
    if existing.empty:
        return new.copy()

    all_rows = pd.concat([existing, new], ignore_index=True)

    valid_keys = [k for k in keys if k in all_rows.columns]
    if not valid_keys:
        return all_rows

    return all_rows.drop_duplicates(valid_keys, keep="first")


def capture(args):
    best = pd.read_csv(args.best_csv)

    if "qualifies" not in best.columns:
        raise ValueError("best CSV has no 'qualifies' column.")

    q = best[best["qualifies"].map(truthy)].copy()

    if q.empty:
        print("No qualifying v3.1 signals to capture.")
        return

    now = pd.Timestamp.now(tz="UTC").isoformat()

    keep = [
        "game_id",
        "season",
        "week",
        "away_team",
        "home_team",
        "market",
        "pick",
        "line",
        "odds",
        "book",
        "quote_updated_at",
        "quote_age_minutes",
        "raw_probability",
        "calibrated_probability",
        "selected_probability",
        "break_even",
        "edge",
        "minimum_edge",
        "model_margin",
        "model_total",
        "away_qb_name",
        "home_qb_name",
    ]
    keep = [c for c in keep if c in q.columns]

    q = q[keep].copy()

    if "season" not in q:
        q["season"] = args.season
    if "week" not in q:
        q["week"] = args.week

    q["entry_snapshot_label"] = args.snapshot_label
    q["captured_utc"] = now
    q["model_version"] = "v3.1"
    q["record_type"] = "production_qualifier"
    q["placed"] = False
    q["stake_units"] = np.nan
    q["status"] = "QUALIFIED_NOT_MARKED_PLACED"

    # Closing / settlement fields.
    for c in [
        "closing_snapshot_label",
        "closing_captured_utc",
        "close_consensus_line",
        "close_best_line",
        "close_best_odds",
        "close_best_book",
        "clv_points_vs_consensus",
        "clv_points_vs_best",
        "same_line_close_best_odds",
        "same_line_price_clv_probability",
        "actual_margin",
        "actual_total",
        "result",
        "settlement_profit_units",
    ]:
        q[c] = np.nan

    ledger_path = Path(args.ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    if ledger_path.exists():
        old = normalize_ledger_dtypes(pd.read_csv(ledger_path))
    else:
        old = pd.DataFrame()

    keys = [
        "game_id",
        "market",
        "pick",
        "line",
        "odds",
        "book",
        "entry_snapshot_label",
    ]

    out = append_unique(old, q, keys)
    out.to_csv(ledger_path, index=False)

    print("\nV3.9 CAPTURED QUALIFIERS")
    show = [
        "game_id", "market", "pick", "line", "odds", "book",
        "selected_probability", "break_even", "edge",
        "model_margin", "model_total", "status",
    ]
    show = [c for c in show if c in q.columns]
    print(q[show].to_string(index=False))
    print(f"\nLedger: {ledger_path}")


def mark(args):
    p = Path(args.ledger)
    if not p.exists():
        raise SystemExit(f"Ledger not found: {p}")

    d = normalize_ledger_dtypes(pd.read_csv(p))

    mask = (
        d["game_id"].astype(str).eq(args.game_id)
        & d["market"].astype(str).str.lower().eq(args.market.lower())
        & d["pick"].astype(str).str.upper().eq(args.pick.upper())
    )

    if not mask.any():
        raise SystemExit("No matching qualifier in ledger.")

    d.loc[mask, "placed"] = True
    d.loc[mask, "stake_units"] = float(args.stake_units)
    d.loc[mask, "status"] = "PLACED"

    d.to_csv(p, index=False)
    print(
        d.loc[
            mask,
            ["game_id", "market", "pick", "line", "odds", "book",
             "placed", "stake_units", "status"],
        ].to_string(index=False)
    )


def _fresh_candidates(candidates):
    d = candidates.copy()
    if "quote_fresh" in d.columns:
        d = d[d["quote_fresh"].map(truthy)].copy()
    return d


def _same_pick_market(candidates, row):
    x = candidates[
        candidates["game_id"].astype(str).eq(str(row["game_id"]))
        & candidates["market"].astype(str).str.lower().eq(
            str(row["market"]).lower()
        )
        & candidates["pick"].astype(str).str.upper().eq(
            str(row["pick"]).upper()
        )
    ].copy()

    x["line"] = pd.to_numeric(x["line"], errors="coerce")
    x["odds"] = pd.to_numeric(x["odds"], errors="coerce")
    return x.dropna(subset=["line", "odds"])


def _point_clv(entry_line, close_line, market, pick):
    entry_line = float(entry_line)
    close_line = float(close_line)
    market = str(market).lower()
    pick = str(pick).upper()

    if market == "spread":
        # Higher handicap is always better for the selected team:
        # +4 > +3, and -2.5 > -3.
        return entry_line - close_line

    if market == "total":
        if pick == "OVER":
            # Lower over line is better.
            return close_line - entry_line
        if pick == "UNDER":
            # Higher under line is better.
            return entry_line - close_line

    return np.nan


def close(args):
    ledger_path = Path(args.ledger)
    if not ledger_path.exists():
        raise SystemExit(f"Ledger not found: {ledger_path}")

    ledger = normalize_ledger_dtypes(pd.read_csv(ledger_path))
    candidates = _fresh_candidates(pd.read_csv(args.all_candidates_csv))

    requested_games = {str(x) for x in args.game_id}
    available_games = set(ledger["game_id"].astype(str))
    missing_games = sorted(requested_games - available_games)
    if missing_games:
        raise SystemExit(
            "Requested game_id not found in ledger: "
            + ", ".join(missing_games)
        )

    now = pd.Timestamp.now(tz="UTC").isoformat()

    updated = 0

    for idx, row in ledger.iterrows():
        if int(row.get("season", args.season)) != args.season:
            continue
        if int(row.get("week", args.week)) != args.week:
            continue
        if str(row["game_id"]) not in requested_games:
            continue

        x = _same_pick_market(candidates, row)
        if x.empty:
            continue

        # Consensus of the quoted line for the exact selected side.
        consensus_line = float(x["line"].median())

        market = str(row["market"]).lower()
        pick = str(row["pick"]).upper()

        # Best closing line for the selected side.
        if market == "spread":
            best_line = float(x["line"].max())
            best_rows = x[x["line"] == best_line]
        elif market == "total" and pick == "OVER":
            best_line = float(x["line"].min())
            best_rows = x[x["line"] == best_line]
        elif market == "total" and pick == "UNDER":
            best_line = float(x["line"].max())
            best_rows = x[x["line"] == best_line]
        else:
            continue

        best_row = best_rows.sort_values("odds", ascending=False).iloc[0]

        entry_line = float(row["line"])
        entry_odds = float(row["odds"])

        ledger.loc[idx, "closing_snapshot_label"] = args.snapshot_label
        ledger.loc[idx, "closing_captured_utc"] = now
        ledger.loc[idx, "close_consensus_line"] = consensus_line
        ledger.loc[idx, "close_best_line"] = best_line
        ledger.loc[idx, "close_best_odds"] = float(best_row["odds"])
        ledger.loc[idx, "close_best_book"] = str(best_row["book"])

        ledger.loc[idx, "clv_points_vs_consensus"] = _point_clv(
            entry_line, consensus_line, market, pick
        )
        ledger.loc[idx, "clv_points_vs_best"] = _point_clv(
            entry_line, best_line, market, pick
        )

        same_line = x[np.isclose(x["line"], entry_line)].copy()
        if not same_line.empty:
            best_same = same_line.sort_values(
                "odds", ascending=False
            ).iloc[0]
            same_odds = float(best_same["odds"])
            ledger.loc[idx, "same_line_close_best_odds"] = same_odds
            ledger.loc[idx, "same_line_price_clv_probability"] = (
                american_break_even(same_odds)
                - american_break_even(entry_odds)
            )

        if str(row.get("status", "")).startswith("QUALIFIED"):
            ledger.loc[idx, "status"] = "CLOSE_CAPTURED"
        elif str(row.get("status", "")) == "PLACED":
            ledger.loc[idx, "status"] = "PLACED_CLOSE_CAPTURED"

        updated += 1

    ledger.to_csv(ledger_path, index=False)

    print(f"Updated closing snapshot for {updated} ledger rows.")
    report_table(ledger, args.season, args.week)


def settle(args):
    ledger_path = Path(args.ledger)
    if not ledger_path.exists():
        raise SystemExit(f"Ledger not found: {ledger_path}")

    ledger = normalize_ledger_dtypes(pd.read_csv(ledger_path))
    results = pd.read_csv(args.results_csv)

    required = ["game_id", "actual_margin", "actual_total"]
    missing = [c for c in required if c not in results.columns]
    if missing:
        raise ValueError(f"Results CSV missing: {missing}")

    results = results[required].drop_duplicates("game_id")
    rmap = results.set_index("game_id").to_dict("index")

    settled = 0

    for idx, row in ledger.iterrows():
        r = rmap.get(str(row["game_id"]))
        if not r:
            continue

        actual_margin = float(r["actual_margin"])
        actual_total = float(r["actual_total"])
        market = str(row["market"]).lower()
        pick = str(row["pick"]).upper()
        line = float(row["line"])
        odds = float(row["odds"])

        if market == "spread":
            home = str(row.get("home_team", "")).upper()
            away = str(row.get("away_team", "")).upper()

            if pick == home:
                value = actual_margin + line
            elif pick == away:
                value = (-actual_margin) + line
            else:
                continue

        elif market == "total":
            if pick == "OVER":
                value = actual_total - line
            elif pick == "UNDER":
                value = line - actual_total
            else:
                continue
        else:
            continue

        result = (
            "win" if value > 0
            else "loss" if value < 0
            else "push"
        )

        stake = row.get("stake_units", np.nan)
        if pd.isna(stake):
            stake = 1.0
        stake = float(stake)

        if result == "win":
            profit = stake * american_profit_per_unit(odds)
        elif result == "loss":
            profit = -stake
        else:
            profit = 0.0

        ledger.loc[idx, "actual_margin"] = actual_margin
        ledger.loc[idx, "actual_total"] = actual_total
        ledger.loc[idx, "result"] = result
        ledger.loc[idx, "settlement_profit_units"] = profit

        if truthy(row.get("placed", False)):
            ledger.loc[idx, "status"] = "PLACED_SETTLED"
        else:
            ledger.loc[idx, "status"] = "SHADOW_SETTLED"

        settled += 1

    ledger.to_csv(ledger_path, index=False)
    print(f"Settled {settled} ledger rows.")
    report_table(ledger, args.season, args.week)


def report_table(d, season=None, week=None):
    x = d.copy()

    if season is not None and "season" in x.columns:
        x = x[pd.to_numeric(x["season"], errors="coerce").eq(season)]
    if week is not None and "week" in x.columns:
        x = x[pd.to_numeric(x["week"], errors="coerce").eq(week)]

    show = [
        "game_id", "market", "pick", "line", "odds", "book",
        "edge", "placed", "stake_units",
        "close_consensus_line", "close_best_line",
        "clv_points_vs_consensus", "clv_points_vs_best",
        "same_line_price_clv_probability",
        "result", "settlement_profit_units", "status",
    ]
    show = [c for c in show if c in x.columns]

    print("\nV3.9 FORWARD-TEST LEDGER")
    if x.empty:
        print("EMPTY")
    else:
        print(x[show].to_string(index=False))


def report(args):
    p = Path(args.ledger)
    if not p.exists():
        raise SystemExit(f"Ledger not found: {p}")
    d = normalize_ledger_dtypes(pd.read_csv(p))
    report_table(d, args.season, args.week)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture")
    cap.add_argument(
        "--best-csv",
        default="artifacts/live/week3_2026_best_v31.csv",
    )
    cap.add_argument("--season", type=int, default=2026)
    cap.add_argument("--week", type=int, default=3)
    cap.add_argument("--snapshot-label", default="current")
    cap.add_argument("--ledger", default=DEFAULT_LEDGER)
    cap.set_defaults(func=capture)

    mk = sub.add_parser("mark")
    mk.add_argument("--ledger", default=DEFAULT_LEDGER)
    mk.add_argument("--game-id", required=True)
    mk.add_argument("--market", choices=["spread", "total"], required=True)
    mk.add_argument("--pick", required=True)
    mk.add_argument("--stake-units", type=float, default=1.0)
    mk.set_defaults(func=mark)

    cl = sub.add_parser("close")
    cl.add_argument(
        "--all-candidates-csv",
        required=True,
    )
    cl.add_argument(
        "--game-id",
        action="append",
        required=True,
        help=(
            "Game to update. Repeat --game-id for games sharing the same "
            "kickoff window. This prevents later games from being mislabeled "
            "as closing snapshots."
        ),
    )
    cl.add_argument("--season", type=int, default=2026)
    cl.add_argument("--week", type=int, default=3)
    cl.add_argument("--snapshot-label", default="closing_candidate")
    cl.add_argument("--ledger", default=DEFAULT_LEDGER)
    cl.set_defaults(func=close)

    st = sub.add_parser("settle")
    st.add_argument("--results-csv", required=True)
    st.add_argument("--season", type=int, default=2026)
    st.add_argument("--week", type=int, default=3)
    st.add_argument("--ledger", default=DEFAULT_LEDGER)
    st.set_defaults(func=settle)

    rp = sub.add_parser("report")
    rp.add_argument("--season", type=int, default=None)
    rp.add_argument("--week", type=int, default=None)
    rp.add_argument("--ledger", default=DEFAULT_LEDGER)
    rp.set_defaults(func=report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
