from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

TEAM_ALIAS = {"JAC": "JAX", "WSH": "WAS", "LAR": "LA"}


def clean(v: Any):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def normalize_market(v: Any) -> str:
    s = str(v or "").strip().lower().replace("_", "").replace("-", "")
    if s in {"ml", "moneyline", "moneylines"}:
        return "moneyline"
    if s in {"spread", "spreads"}:
        return "spread"
    if s in {"total", "totals", "ou", "overunder"}:
        return "total"
    return s


def normalize_line(market: str, value: Any):
    x = clean(value)
    if market == "moneyline" and x is None:
        return 0.0
    return x


def normalize_team(team: str) -> str:
    team = str(team or "").upper().strip()
    return TEAM_ALIAS.get(team, team)


def same_num(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def parse_teams(game_id: str):
    p = str(game_id).split("_")
    return (p[-2], p[-1]) if len(p) >= 4 else ("AWAY", "HOME")


def load_csv(path: str | None):
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_snapshot(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    marker = "export const APP_DATA = "
    if marker not in text:
        return {}
    payload = text.split(marker, 1)[1]
    suffix = " satisfies AppSnapshot;"
    if suffix in payload:
        payload = payload.rsplit(suffix, 1)[0]
    return json.loads(payload.strip().rstrip(";"))


def fetch_schedule(season: int, week: int):
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        f"?dates={season}&seasontype=2&week={week}&limit=100"
    )
    req = Request(url, headers={"User-Agent": "NFL-SIM/5.1"})
    try:
        with urlopen(req, timeout=15) as resp:
            payload = json.load(resp)
    except Exception as exc:
        print(f"Schedule lookup warning for Week {week}: {exc}")
        return {}, []

    schedule = {}
    results = []
    for event in payload.get("events", []):
        comps = event.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors") or []
        home = next((x for x in competitors if x.get("homeAway") == "home"), None)
        away = next((x for x in competitors if x.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        home_abbr = normalize_team((home.get("team") or {}).get("abbreviation"))
        away_abbr = normalize_team((away.get("team") or {}).get("abbreviation"))
        if not home_abbr or not away_abbr:
            continue

        game_id = f"{season}_{week:02d}_{away_abbr}_{home_abbr}"
        schedule[game_id] = event.get("date") or comp.get("date")

        completed = bool((((comp.get("status") or {}).get("type") or {}).get("completed")))
        if not completed:
            continue
        try:
            away_score = float(away.get("score"))
            home_score = float(home.get("score"))
        except Exception:
            continue
        results.append({
            "gameId": game_id,
            "actualMargin": home_score - away_score,
            "actualTotal": home_score + away_score,
            "awayScore": away_score,
            "homeScore": home_score,
        })
    return schedule, results


def build_odds_board(all_candidates: pd.DataFrame, best_row):
    if best_row is None or all_candidates.empty:
        return []
    required = {"game_id", "market", "pick", "book", "line", "odds"}
    if not required.issubset(all_candidates.columns):
        return []

    gid = str(best_row.get("game_id"))
    market = normalize_market(best_row.get("market"))
    pick = str(best_row.get("pick", ""))

    x = all_candidates.copy()
    x["_market_norm"] = x["market"].map(normalize_market)
    x = x[
        x["game_id"].astype(str).eq(gid)
        & x["_market_norm"].eq(market)
        & x["pick"].astype(str).eq(pick)
    ].copy()
    if x.empty:
        return []

    if "edge" in x.columns:
        x["_edge_sort"] = pd.to_numeric(x["edge"], errors="coerce").fillna(-999.0)
        x = x.sort_values("_edge_sort", ascending=False)
    x = x.drop_duplicates(subset=["book", "line", "odds"], keep="first")

    options = []
    best_line = normalize_line(market, best_row.get("line"))
    for _, r in x.iterrows():
        selected = (
            str(r.get("book", "")) == str(best_row.get("book", ""))
            and same_num(normalize_line(market, r.get("line")), best_line)
            and same_num(r.get("odds"), best_row.get("odds"))
        )
        options.append({
            "line": normalize_line(market, r.get("line")),
            "odds": clean(r.get("odds")),
            "book": str(r.get("book", "")),
            "breakEven": clean(r.get("break_even")),
            "edge": clean(r.get("edge")),
            "quoteAgeMinutes": clean(r.get("quote_age_minutes")),
            "selected": selected,
        })
    options.sort(key=lambda q: (
        not q["selected"],
        -(q["edge"] if q["edge"] is not None else -999.0),
        q["book"],
    ))
    return options[:8]


def market_pick(r, all_candidates):
    if r is None:
        return None
    market = normalize_market(r.get("market"))
    return {
        "market": market,
        "pick": str(r.get("pick", "")),
        "line": normalize_line(market, r.get("line")),
        "odds": clean(r.get("odds")),
        "book": str(r.get("book", "")),
        "rawProbability": clean(r.get("raw_probability")),
        "calibratedProbability": clean(r.get("calibrated_probability")),
        "selectedProbability": clean(r.get("selected_probability")),
        "breakEven": clean(r.get("break_even")),
        "edge": clean(r.get("edge")),
        "minimumEdge": clean(r.get("minimum_edge")),
        "qualifies": truthy(r.get("qualifies")),
        "quoteAgeMinutes": clean(r.get("quote_age_minutes")),
        "oddsBoard": build_odds_board(all_candidates, r),
    }


def one_market(df: pd.DataFrame, market: str):
    if df.empty or "market" not in df.columns:
        return None
    x = df[df["market"].map(normalize_market).eq(market)]
    return None if x.empty else x.iloc[0]


def kickoff_from_row(row, game_id: str, schedule: dict[str, str | None]):
    if row is not None:
        for col in ("kickoff_at", "commence_time", "kickoff", "start_time", "game_time"):
            value = clean(row.get(col)) if hasattr(row, "get") else None
            if value:
                try:
                    ts = pd.Timestamp(value)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("America/New_York")
                    return ts.isoformat()
                except Exception:
                    pass
    return schedule.get(str(game_id))


def build_games(best: pd.DataFrame, all_candidates: pd.DataFrame, season: int, week: int, schedule):
    if best.empty:
        return []
    games = []
    for game_id, g in best.groupby("game_id"):
        spread_r = one_market(g, "spread")
        total_r = one_market(g, "total")
        moneyline_r = one_market(g, "moneyline")
        ex = spread_r if spread_r is not None else total_r if total_r is not None else moneyline_r
        away, home = parse_teams(game_id)
        spread = market_pick(spread_r, all_candidates)
        total = market_pick(total_r, all_candidates)
        moneyline = market_pick(moneyline_r, all_candidates)
        ps = [p for p in (spread, total, moneyline) if p]
        edges = [p["edge"] for p in ps if p["edge"] is not None]
        games.append({
            "id": str(game_id),
            "season": season,
            "week": week,
            "away": away,
            "home": home,
            "kickoffAt": kickoff_from_row(ex, str(game_id), schedule),
            "awayQB": clean(ex.get("away_qb_name")) if ex is not None else None,
            "homeQB": clean(ex.get("home_qb_name")) if ex is not None else None,
            "modelMargin": clean(ex.get("model_margin")) if ex is not None else None,
            "modelTotal": clean(ex.get("model_total")) if ex is not None else None,
            "spread": spread,
            "total": total,
            "moneyline": moneyline,
            "shadowTotal": None,
            "qualifies": any(p["qualifies"] for p in ps),
            "bestEdge": max(edges) if edges else None,
        })
    games.sort(key=lambda x: (x["kickoffAt"] or "9999", x["id"]))
    return games


def merge_results(existing, updates):
    merged = {str(x.get("gameId")): x for x in (existing or []) if x.get("gameId")}
    for x in updates:
        if x.get("gameId"):
            merged[str(x["gameId"])] = x
    return list(merged.values())


def default_performance():
    return {
        "qualifiedCount": 0, "settledCount": 0, "wins": 0, "losses": 0,
        "pushes": 0, "forwardUnits": 0, "placedCount": 0,
        "clvCount": 0, "positiveClvCount": 0, "averageClvPoints": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--current-week", type=int, required=True)
    ap.add_argument("--next-week", type=int, default=0)
    ap.add_argument("--current-best")
    ap.add_argument("--current-all-candidates")
    ap.add_argument("--next-best")
    ap.add_argument("--next-all-candidates")
    ap.add_argument("--current-snapshot", default="mobile/generatedData.ts")
    ap.add_argument("--output", default="mobile/generatedData.ts")
    args = ap.parse_args()

    previous = load_snapshot(args.current_snapshot)
    current_schedule, current_results = fetch_schedule(args.season, args.current_week)
    previous_results = []
    if args.current_week > 1:
        _, previous_results = fetch_schedule(args.season, args.current_week - 1)

    current_best = load_csv(args.current_best)
    current_candidates = load_csv(args.current_all_candidates)
    if not current_best.empty:
        current_games = build_games(current_best, current_candidates, args.season, args.current_week, current_schedule)
    else:
        current_games = [g for g in previous.get("games", []) if int(g.get("week", -1)) == args.current_week]

    if not current_games:
        raise RuntimeError(f"No Week {args.current_week} model games are available.")

    next_games = []
    if args.next_week:
        next_schedule, _ = fetch_schedule(args.season, args.next_week)
        next_best = load_csv(args.next_best)
        next_candidates = load_csv(args.next_all_candidates)
        if not next_best.empty:
            next_games = build_games(next_best, next_candidates, args.season, args.next_week, next_schedule)
        else:
            next_games = [g for g in previous.get("games", []) if int(g.get("week", -1)) == args.next_week]

    games = current_games + next_games
    games.sort(key=lambda x: (int(x.get("week", 999)), x.get("kickoffAt") or "9999", x.get("id") or ""))
    available = [args.current_week]
    if next_games and args.next_week:
        available.append(args.next_week)

    game_results = merge_results(previous.get("gameResults", []), previous_results + current_results)
    data = {
        "generatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "season": args.season,
        "week": args.current_week,
        "currentWeek": args.current_week,
        "nextWeek": args.next_week or None,
        "availableWeeks": available,
        "games": games,
        "gameResults": game_results,
        "forwardPicks": previous.get("forwardPicks", []),
        "historicalMetrics": previous.get("historicalMetrics", []),
        "performance": previous.get("performance", default_performance()),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        'import type { AppSnapshot } from "./types";\n\n'
        + "export const APP_DATA = "
        + json.dumps(data, indent=2)
        + " satisfies AppSnapshot;\n",
        encoding="utf-8",
    )

    print("V5.1 AUTO-ROLL MOBILE EXPORT")
    for week in available:
        week_games = [g for g in games if int(g.get("week", -1)) == week]
        q = sum(1 for g in week_games if g.get("qualifies"))
        print(f"Week {week}: {len(week_games)} games, {q} games with qualifying market(s)")
    print(f"Stored final game results: {len(game_results)}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
