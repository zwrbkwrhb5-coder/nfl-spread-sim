from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .live_inputs_v30 import (
    FULL_NAME_TO_TEAM,
    harden_market,
    load_schedule,
    normalize_team,
)
from .live_raw_qb_v29 import (
    SPREAD_FEATURES,
    TOTAL_FEATURES,
    american_break_even,
    build_consistent_team_state,
    build_live_context,
    build_live_features,
    build_raw_qb_state,
    fit_final_isotonic,
    load_current_pbp,
    load_required_params,
)


ODDS_API_URL = (
    "https://api.the-odds-api.com/v4/sports/"
    "americanfootball_nfl/odds/"
)

# Indiana Gaming Commission mobile books.
# Keys on the left are our canonical labels.
INDIANA_MOBILE_BOOKS = {
    "ballybet": "Bally Bet",
    "bet365": "bet365",
    "betmgm": "BetMGM",
    "betrivers": "BetRivers",
    "caesars": "Caesars",
    "draftkings": "DraftKings",
    "fanatics": "Fanatics",
    "fanduel": "FanDuel",
    "hardrockbet": "Hard Rock Bet",
    "sbk": "SBK",
    "thescore": "theScore Bet",
}

# Current The Odds API keys for Indiana books it exposes.
ODDS_API_TO_INDIANA = {
    "ballybet": "ballybet",
    "betmgm": "betmgm",
    "betrivers": "betrivers",
    "williamhill_us": "caesars",
    "draftkings": "draftkings",
    "fanatics": "fanatics",
    "fanduel": "fanduel",
    "hardrockbet": "hardrockbet",
    "espnbet": "thescore",
}

KNOWN_API_GAPS = {"bet365", "sbk"}


def _json_url(url: str, timeout: int = 25):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "nfl-spread-sim-v3.1"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8")
        headers = dict(r.headers.items())
    return json.loads(body), headers


def fetch_indiana_nfl_odds(api_key: str):
    params = {
        "apiKey": api_key,
        "regions": "us,us2",
        "markets": "spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    url = ODDS_API_URL + "?" + urllib.parse.urlencode(params)
    return _json_url(url)


def full_name_to_abbr(name):
    return normalize_team(FULL_NAME_TO_TEAM.get(name, name))


def parse_quotes(events) -> pd.DataFrame:
    rows = []

    for event in events:
        home = full_name_to_abbr(event.get("home_team"))
        away = full_name_to_abbr(event.get("away_team"))
        commence = event.get("commence_time")

        if not home or not away:
            continue

        for book in event.get("bookmakers", []):
            api_key = book.get("key")
            canonical = ODDS_API_TO_INDIANA.get(api_key)

            if canonical is None:
                continue

            book_title = INDIANA_MOBILE_BOOKS[canonical]
            book_updated = book.get("last_update")

            for market in book.get("markets", []):
                market_key = market.get("key")
                market_updated = market.get("last_update") or book_updated

                if market_key == "spreads":
                    for o in market.get("outcomes", []):
                        team = full_name_to_abbr(o.get("name"))
                        if team not in {home, away}:
                            continue

                        rows.append({
                            "event_id": event.get("id"),
                            "commence_time": commence,
                            "away_team": away,
                            "home_team": home,
                            "book_key": api_key,
                            "book": book_title,
                            "indiana_book_key": canonical,
                            "market": "spread",
                            "side": "home" if team == home else "away",
                            "pick": team,
                            "line": o.get("point"),
                            "odds": o.get("price"),
                            "quote_updated_at": market_updated,
                        })

                elif market_key == "totals":
                    for o in market.get("outcomes", []):
                        name = str(o.get("name", "")).strip().lower()
                        if name not in {"over", "under"}:
                            continue

                        rows.append({
                            "event_id": event.get("id"),
                            "commence_time": commence,
                            "away_team": away,
                            "home_team": home,
                            "book_key": api_key,
                            "book": book_title,
                            "indiana_book_key": canonical,
                            "market": "total",
                            "side": name,
                            "pick": name.upper(),
                            "line": o.get("point"),
                            "odds": o.get("price"),
                            "quote_updated_at": market_updated,
                        })

    q = pd.DataFrame(rows)
    if q.empty:
        return q

    q["line"] = pd.to_numeric(q["line"], errors="coerce")
    q["odds"] = pd.to_numeric(q["odds"], errors="coerce")
    q["quote_updated_at"] = pd.to_datetime(
        q["quote_updated_at"],
        errors="coerce",
        utc=True,
    )

    return q.dropna(subset=["line", "odds"]).reset_index(drop=True)


def add_quote_freshness(
    quotes: pd.DataFrame,
    max_age_minutes: float,
) -> pd.DataFrame:
    q = quotes.copy()

    now = pd.Timestamp.now(tz="UTC")
    age = (
        now - pd.to_datetime(q["quote_updated_at"], utc=True)
    ).dt.total_seconds() / 60.0

    q["quote_age_minutes"] = age
    q["quote_fresh"] = (
        q["quote_updated_at"].notna()
        & (age >= -5)
        & (age <= float(max_age_minutes))
    )
    return q


def attach_schedule_game_ids(
    quotes: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    s = schedule.copy()
    s["home_team"] = s["home_team"].map(normalize_team)
    s["away_team"] = s["away_team"].map(normalize_team)

    keep = [
        c for c in [
            "game_id", "season", "week",
            "away_team", "home_team",
        ]
        if c in s.columns
    ]

    return quotes.merge(
        s[keep].drop_duplicates(["away_team", "home_team"]),
        on=["away_team", "home_team"],
        how="inner",
    )


def build_base_market(
    quotes: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    """
    One row/game only. Lines here are placeholders for feature construction.
    Every actual candidate line is scored later from `quotes`.
    """
    rows = []

    for game_id, g in quotes.groupby("game_id"):
        g = g.sort_values("quote_updated_at", ascending=False)
        r = g.iloc[0]

        spread = g[g["market"] == "spread"]
        total = g[g["market"] == "total"]

        hs = spread[spread["side"] == "home"]
        as_ = spread[spread["side"] == "away"]
        ov = total[total["side"] == "over"]
        un = total[total["side"] == "under"]

        rows.append({
            "game_id": game_id,
            "season": season,
            "week": week,
            "away_team": r["away_team"],
            "home_team": r["home_team"],
            "home_spread": (
                float(hs.iloc[0]["line"]) if not hs.empty else 0.0
            ),
            "total_line": (
                float(ov.iloc[0]["line"]) if not ov.empty else 45.0
            ),
            "home_spread_odds": (
                float(hs.iloc[0]["odds"]) if not hs.empty else -110.0
            ),
            "away_spread_odds": (
                float(as_.iloc[0]["odds"]) if not as_.empty else -110.0
            ),
            "over_odds": (
                float(ov.iloc[0]["odds"]) if not ov.empty else -110.0
            ),
            "under_odds": (
                float(un.iloc[0]["odds"]) if not un.empty else -110.0
            ),
            "market_updated_at": r["quote_updated_at"],
            "market_source": "the_odds_api_multibook",
        })

    return pd.DataFrame(rows)


def fit_live_models(train: pd.DataFrame):
    spread_train = train.dropna(
        subset=SPREAD_FEATURES + ["actual_margin"]
    )
    total_train = train.dropna(
        subset=TOTAL_FEATURES + ["actual_total"]
    )

    spread_model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    total_model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])

    spread_model.fit(
        spread_train[SPREAD_FEATURES],
        spread_train["actual_margin"],
    )
    total_model.fit(
        total_train[TOTAL_FEATURES],
        total_train["actual_total"],
    )

    return spread_model, total_model


def make_projections(
    hardened_market: pd.DataFrame,
    training_csv: str,
    team_games: str,
    qb_games: str,
    season: int,
    week: int,
) -> pd.DataFrame:
    train = pd.read_csv(training_csv)
    current_pbp = load_current_pbp(season)

    team_state = build_consistent_team_state(
        team_games,
        current_pbp,
        season,
        week,
    )

    qb_state = build_raw_qb_state(
        qb_games,
        current_pbp,
        hardened_market,
        season,
        week,
    )

    context = build_live_context(
        hardened_market,
        season,
        week,
    )

    live = build_live_features(
        hardened_market,
        team_state,
        qb_state,
        context,
    )

    spread_model, total_model = fit_live_models(train)

    live["spread_feature_ready_v31"] = (
        ~live[SPREAD_FEATURES].isna().any(axis=1)
    )
    live["total_feature_ready_v31"] = (
        ~live[TOTAL_FEATURES].isna().any(axis=1)
    )

    live["model_margin"] = np.nan
    live["model_total"] = np.nan

    sm = live["spread_feature_ready_v31"]
    tm = live["total_feature_ready_v31"]

    live.loc[sm, "model_margin"] = spread_model.predict(
        live.loc[sm, SPREAD_FEATURES]
    )
    live.loc[tm, "model_total"] = total_model.predict(
        live.loc[tm, TOTAL_FEATURES]
    )

    # Preserve verified v3.0 QB metadata from the unique game input rather
    # than relying on the v2.9 merge's duplicate/suffixed metadata columns.
    meta_cols = [
        "game_id",
        "home_qb_name",
        "away_qb_name",
        "home_qb_input_source",
        "away_qb_input_source",
        "spread_input_ready_v30",
        "total_input_ready_v30",
        "roof_type_v30",
        "weather_source_v30",
        "temp",
        "wind",
    ]
    meta_cols = [
        c for c in meta_cols if c in hardened_market.columns
    ]

    meta = hardened_market[meta_cols].drop_duplicates("game_id")
    for c in meta_cols:
        if c != "game_id" and c in live.columns:
            live = live.drop(columns=[c])

    live = live.merge(meta, on="game_id", how="left")

    return live


def empirical_gt(residuals, threshold: float) -> float:
    x = np.asarray(residuals, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x > threshold))


def empirical_lt(residuals, threshold: float) -> float:
    x = np.asarray(residuals, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x < threshold))


def score_quotes(
    quotes: pd.DataFrame,
    projections: pd.DataFrame,
    oos: pd.DataFrame,
    calibrated_sides: pd.DataFrame,
    total_params: dict,
    spread_params: dict,
) -> pd.DataFrame:
    pcols = [
        "game_id",
        "model_margin",
        "model_total",
        "spread_feature_ready_v31",
        "total_feature_ready_v31",
        "spread_input_ready_v30",
        "total_input_ready_v30",
        "home_qb_name",
        "away_qb_name",
        "home_qb_input_source",
        "away_qb_input_source",
        "roof_type_v30",
        "weather_source_v30",
        "temp",
        "wind",
    ]
    pcols = [c for c in pcols if c in projections.columns]

    q = quotes.merge(
        projections[pcols].drop_duplicates("game_id"),
        on="game_id",
        how="left",
    )

    margin_resid = (
        pd.to_numeric(oos["actual_margin"], errors="coerce")
        - pd.to_numeric(oos["pred_margin"], errors="coerce")
    ).dropna().to_numpy(float)

    total_resid = (
        pd.to_numeric(oos["actual_total"], errors="coerce")
        - pd.to_numeric(oos["pred_total"], errors="coerce")
    ).dropna().to_numpy(float)

    spread_iso = fit_final_isotonic(calibrated_sides, "spread")
    total_iso = fit_final_isotonic(calibrated_sides, "total")

    scored = []

    for _, r in q.iterrows():
        market = r["market"]
        side = r["side"]
        line = float(r["line"])
        odds = float(r["odds"])

        if market == "spread":
            if not bool(r.get("spread_input_ready_v30", False)):
                continue
            if not bool(r.get("spread_feature_ready_v31", False)):
                continue
            if pd.isna(r.get("model_margin")):
                continue

            model_margin = float(r["model_margin"])

            if side == "home":
                # home covers when actual home margin + home spread > 0
                threshold = -(model_margin + line)
                raw_p = empirical_gt(margin_resid, threshold)
            else:
                # away covers when actual home margin < away handicap
                threshold = line - model_margin
                raw_p = empirical_lt(margin_resid, threshold)

            calibrated_p = float(spread_iso.predict([raw_p])[0])
            lam = float(
                spread_params["probability_shrinkage_lambda"]
            )
            min_edge = float(spread_params["minimum_edge"])

        elif market == "total":
            if not bool(r.get("total_input_ready_v30", False)):
                continue
            if not bool(r.get("total_feature_ready_v31", False)):
                continue
            if pd.isna(r.get("model_total")):
                continue

            model_total = float(r["model_total"])
            threshold = line - model_total

            if side == "over":
                raw_p = empirical_gt(total_resid, threshold)
            else:
                raw_p = empirical_lt(total_resid, threshold)

            calibrated_p = float(total_iso.predict([raw_p])[0])
            lam = float(
                total_params["probability_shrinkage_lambda"]
            )
            min_edge = float(total_params["minimum_edge"])

        else:
            continue

        selected_p = 0.5 + lam * (calibrated_p - 0.5)
        break_even = american_break_even(odds)
        edge = selected_p - break_even

        d = r.to_dict()
        d.update({
            "raw_probability": raw_p,
            "calibrated_probability": calibrated_p,
            "selected_probability": selected_p,
            "break_even": break_even,
            "edge": edge,
            "minimum_edge": min_edge,
            "qualifies": (
                bool(r.get("quote_fresh", False))
                and edge >= min_edge
            ),
        })
        scored.append(d)

    out = pd.DataFrame(scored)
    if out.empty:
        return out

    return out.sort_values(
        ["qualifies", "edge"],
        ascending=[False, False],
    ).reset_index(drop=True)


def select_best_per_game_market(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()

    fresh = scored[scored["quote_fresh"]].copy()
    if fresh.empty:
        return fresh

    # Best actual price/line combination according to calibrated edge.
    best = (
        fresh.sort_values(
            ["game_id", "market", "edge", "quote_updated_at"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(["game_id", "market"], keep="first")
        .sort_values(
            ["qualifies", "edge"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )
    return best


def archive_snapshot(
    quotes: pd.DataFrame,
    snapshot_dir: str,
) -> Path:
    p = Path(snapshot_dir)
    p.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    out = p / f"odds_v31_{stamp}.csv"
    quotes.to_csv(out, index=False)
    return out


def coverage_report(quotes: pd.DataFrame):
    returned = set(
        quotes["indiana_book_key"].dropna().astype(str)
        if not quotes.empty
        else []
    )
    expected = set(INDIANA_MOBILE_BOOKS)
    api_mapped = set(ODDS_API_TO_INDIANA.values())

    print("\nV3.1 INDIANA SPORTSBOOK COVERAGE")
    for key, title in INDIANA_MOBILE_BOOKS.items():
        if key in returned:
            status = "CURRENT ODDS RETURNED"
        elif key in KNOWN_API_GAPS:
            status = "NOT LISTED BY THE ODDS API"
        elif key in api_mapped:
            status = "SUPPORTED, NO CURRENT NFL QUOTE RETURNED"
        else:
            status = "UNMAPPED"
        print(f"{title:18s}  {status}")

    print(
        f"\nBooks with current returned quotes: {len(returned)} "
        f"/ {len(expected)} Indiana mobile books"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--training-csv",
        default="artifacts/dual_market/training_games_qb_context_v24.csv",
    )
    ap.add_argument(
        "--oos-csv",
        default="artifacts/dual_market/oos_qb_raw_context_v28.csv",
    )
    ap.add_argument(
        "--team-games",
        default="artifacts/team_games.parquet",
    )
    ap.add_argument(
        "--qb-games",
        default="artifacts/dual_market/qb_games_v17.csv",
    )
    ap.add_argument(
        "--calibrated-sides",
        default="artifacts/calibration_v28_raw/calibrated_sides.csv",
    )
    ap.add_argument(
        "--total-params",
        default="artifacts/calibration_v28_raw/latest_params.json",
    )
    ap.add_argument(
        "--spread-params",
        default="artifacts/spread_stability_v28_raw/latest_params.json",
    )
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=3)
    ap.add_argument("--max-age-minutes", type=float, default=30.0)
    ap.add_argument(
        "--snapshot-dir",
        default="artifacts/live/odds_snapshots",
    )
    ap.add_argument(
        "--all-candidates-output",
        default="artifacts/live/week3_2026_all_candidates_v31.csv",
    )
    ap.add_argument(
        "--best-output",
        default="artifacts/live/week3_2026_best_v31.csv",
    )

    args = ap.parse_args()

    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        raise SystemExit(
            "THE_ODDS_API_KEY is not set. "
            "Export it in the terminal or add it as a Codespaces secret. "
            "Do not put the key in source control."
        )

    print("Fetching current NFL odds from Indiana-compatible books...")
    events, response_headers = fetch_indiana_nfl_odds(api_key)

    quotes = parse_quotes(events)
    if quotes.empty:
        raise SystemExit("No Indiana-compatible NFL spread/total quotes returned.")

    quotes = add_quote_freshness(
        quotes,
        max_age_minutes=args.max_age_minutes,
    )

    schedule = load_schedule(args.season, args.week)
    quotes = attach_schedule_game_ids(quotes, schedule)

    if quotes.empty:
        raise SystemExit(
            "Odds were returned, but none matched the requested NFL week."
        )

    snapshot = archive_snapshot(quotes, args.snapshot_dir)
    print(f"Archived raw odds snapshot: {snapshot}")

    coverage_report(quotes)

    base = build_base_market(
        quotes,
        season=args.season,
        week=args.week,
    )

    hardened = harden_market(
        base,
        season=args.season,
        week=args.week,
        max_age_minutes=args.max_age_minutes,
        refresh_odds=False,
    )

    projections = make_projections(
        hardened_market=hardened,
        training_csv=args.training_csv,
        team_games=args.team_games,
        qb_games=args.qb_games,
        season=args.season,
        week=args.week,
    )

    oos = pd.read_csv(args.oos_csv)
    calibration = pd.read_csv(args.calibrated_sides)

    total_params, spread_params = load_required_params(
        args.total_params,
        args.spread_params,
    )

    scored = score_quotes(
        quotes=quotes,
        projections=projections,
        oos=oos,
        calibrated_sides=calibration,
        total_params=total_params,
        spread_params=spread_params,
    )

    if scored.empty:
        raise SystemExit(
            "No quotes survived v3.1 input/model readiness gates."
        )

    best = select_best_per_game_market(scored)

    all_out = Path(args.all_candidates_output)
    all_out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(all_out, index=False)

    best_out = Path(args.best_output)
    best_out.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(best_out, index=False)

    print("\nTHE ODDS API USAGE")
    for h in [
        "x-requests-remaining",
        "x-requests-used",
        "x-requests-last",
    ]:
        value = (
            response_headers.get(h)
            or response_headers.get(h.title())
        )
        if value is not None:
            print(f"{h}: {value}")

    show = [
        "game_id",
        "market",
        "pick",
        "line",
        "odds",
        "book",
        "quote_age_minutes",
        "raw_probability",
        "calibrated_probability",
        "selected_probability",
        "break_even",
        "edge",
        "minimum_edge",
        "qualifies",
        "model_margin",
        "model_total",
        "away_qb_name",
        "home_qb_name",
    ]
    show = [c for c in show if c in best.columns]

    print("\nV3.1 BEST AVAILABLE INDIANA PRICES")
    print(best[show].head(30).to_string(index=False))

    qualifying = best[best["qualifies"]]

    print("\nV3.1 QUALIFYING BETS")
    if qualifying.empty:
        print("NONE")
    else:
        print(
            qualifying[show]
            .head(10)
            .to_string(index=False)
        )

    print(f"\nSaved all candidates: {all_out}")
    print(f"Saved best per game/market: {best_out}")


if __name__ == "__main__":
    main()
