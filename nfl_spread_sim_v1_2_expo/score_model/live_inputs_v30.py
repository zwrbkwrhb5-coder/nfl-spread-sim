from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


NFLVERSE_GAMES_URL = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)

DEPTH_CHART_URLS = [
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "depth_charts/depth_charts_{season}.csv",
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "depth_charts/depth_charts_{season}.parquet",
]

ODDS_API_URL = (
    "https://api.the-odds-api.com/v4/sports/"
    "americanfootball_nfl/odds/"
)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


TEAM_ALIASES = {
    "LAR": "LA",
    "STL": "LA",
    "JAC": "JAX",
    "WSH": "WAS",
    "OAK": "LV",
    "SD": "LAC",
}

FULL_NAME_TO_TEAM = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

# Home venue coordinates. Neutral/international games should provide
# stadium_lat/stadium_lon overrides in the market CSV.
STADIUMS = {
    "ARI": (33.5276, -112.2626),
    "ATL": (33.7554, -84.4008),
    "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7868),
    "CAR": (35.2258, -80.8528),
    "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160),
    "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456),
    "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107),
    "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839),
    "LA": (33.9535, -118.3392),
    "LAC": (33.9535, -118.3392),
    "LV": (36.0908, -115.1830),
    "MIA": (25.9580, -80.2389),
    "MIN": (44.9738, -93.2581),
    "NE": (42.0909, -71.2643),
    "NO": (29.9511, -90.0812),
    "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745),
    "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316),
    "SF": (37.4033, -121.9694),
    "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713),
    "WAS": (38.9076, -76.8645),
}

FIXED_ROOF = {
    "DET": "dome",
    "LV": "dome",
    "MIN": "dome",
    "NO": "dome",
    "LA": "dome",
    "LAC": "dome",
}

RETRACTABLE_ROOF = {"ARI", "ATL", "DAL", "HOU", "IND"}

OUTDOOR_ROOF = {
    t: "outdoors"
    for t in STADIUMS
    if t not in FIXED_ROOF and t not in RETRACTABLE_ROOF
}


def normalize_team(team):
    if pd.isna(team):
        return team
    x = str(team).strip().upper()
    return TEAM_ALIASES.get(x, x)


def normalize_market_teams(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["home_team"] = d["home_team"].map(normalize_team)
    d["away_team"] = d["away_team"].map(normalize_team)

    if "game_id" in d.columns:
        s = d["game_id"].astype(str)
        for old, new in TEAM_ALIASES.items():
            s = s.str.replace(f"_{old}_", f"_{new}_", regex=False)
            s = s.str.replace(f"_{old}$", f"_{new}", regex=True)
        d["game_id"] = s
    return d


def read_json_url(url: str, timeout: int = 20):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "nfl-spread-sim-v3.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def load_schedule(season: int, week: int) -> pd.DataFrame:
    g = pd.read_csv(NFLVERSE_GAMES_URL)
    g = g[
        (pd.to_numeric(g["season"], errors="coerce") == season)
        & (pd.to_numeric(g["week"], errors="coerce") == week)
    ].copy()

    if "home_team" in g:
        g["home_team"] = g["home_team"].map(normalize_team)
    if "away_team" in g:
        g["away_team"] = g["away_team"].map(normalize_team)

    return g


def load_depth_charts(season: int) -> pd.DataFrame:
    errors = []

    for template in DEPTH_CHART_URLS:
        url = template.format(season=season)
        try:
            if url.endswith(".csv"):
                d = pd.read_csv(url)
            else:
                d = pd.read_parquet(url)
            return d
        except Exception as e:
            errors.append(f"{url}: {e}")

    print("WARNING: current depth charts unavailable.")
    for e in errors:
        print(" ", e)
    return pd.DataFrame()


def parse_depth_chart_qbs(depth: pd.DataFrame) -> pd.DataFrame:
    """
    Supports nflverse 2025+ depth-chart schema:
      dt, team, player_name, gsis_id, pos_abb, pos_rank, ...

    Returns one most-recent QB1 per team.
    """
    if depth.empty:
        return pd.DataFrame(
            columns=[
                "team", "qb_id", "qb_name",
                "depth_chart_dt", "qb_source",
            ]
        )

    d = depth.copy()

    team_col = "team" if "team" in d.columns else "club_code"
    name_col = (
        "player_name"
        if "player_name" in d.columns
        else "full_name"
        if "full_name" in d.columns
        else "football_name"
    )
    id_col = "gsis_id" if "gsis_id" in d.columns else None

    if "pos_abb" in d.columns:
        qb = d["pos_abb"].astype(str).str.upper().eq("QB")
    elif "position" in d.columns:
        qb = d["position"].astype(str).str.upper().eq("QB")
    elif "pos_name" in d.columns:
        qb = d["pos_name"].astype(str).str.contains(
            "quarterback", case=False, na=False
        )
    else:
        raise ValueError("Depth chart has no recognizable position column.")

    d = d[qb].copy()
    d["team"] = d[team_col].map(normalize_team)

    if "dt" in d.columns:
        d["depth_chart_dt"] = pd.to_datetime(
            d["dt"], errors="coerce", utc=True
        )
    else:
        d["depth_chart_dt"] = pd.NaT

    if "pos_rank" in d.columns:
        d["_rank"] = pd.to_numeric(d["pos_rank"], errors="coerce")
    elif "depth_team" in d.columns:
        d["_rank"] = pd.to_numeric(d["depth_team"], errors="coerce")
    else:
        d["_rank"] = 99

    rows = []
    for team, g in d.groupby("team"):
        if g["depth_chart_dt"].notna().any():
            latest_dt = g["depth_chart_dt"].max()
            g = g[g["depth_chart_dt"] == latest_dt]

        g = g.sort_values(["_rank"], na_position="last")
        if g.empty:
            continue

        r = g.iloc[0]
        rows.append({
            "team": team,
            "qb_id": (
                str(r[id_col])
                if id_col and pd.notna(r[id_col])
                else np.nan
            ),
            "qb_name": (
                str(r[name_col])
                if name_col in r and pd.notna(r[name_col])
                else np.nan
            ),
            "depth_chart_dt": r["depth_chart_dt"],
            "qb_source": "nflverse_depth_chart",
        })

    return pd.DataFrame(rows)


def add_qb_inputs(
    market: pd.DataFrame,
    depth_qbs: pd.DataFrame,
) -> pd.DataFrame:
    d = market.copy()
    lookup = (
        depth_qbs.set_index("team").to_dict("index")
        if not depth_qbs.empty
        else {}
    )

    for side in ["home", "away"]:
        id_col = f"{side}_qb_id"
        name_col = f"{side}_qb_name"
        source_col = f"{side}_qb_input_source"
        verified_col = f"{side}_qb_verified"
        dt_col = f"{side}_qb_depth_chart_dt"

        if id_col not in d:
            d[id_col] = np.nan
        if name_col not in d:
            d[name_col] = np.nan

        sources = []
        verified = []
        dts = []
        ids = []
        names = []

        for _, r in d.iterrows():
            manual_id = r.get(id_col, np.nan)
            manual_name = r.get(name_col, np.nan)

            has_manual = (
                (pd.notna(manual_id) and str(manual_id).strip())
                or (pd.notna(manual_name) and str(manual_name).strip())
            )

            if has_manual:
                ids.append(manual_id)
                names.append(manual_name)
                sources.append("manual_market_override")
                verified.append(True)
                dts.append(pd.NaT)
                continue

            team = r[f"{side}_team"]
            q = lookup.get(team)

            if q:
                ids.append(q.get("qb_id", np.nan))
                names.append(q.get("qb_name", np.nan))
                sources.append(q.get("qb_source", "nflverse_depth_chart"))
                verified.append(
                    pd.notna(q.get("qb_id"))
                    or pd.notna(q.get("qb_name"))
                )
                dts.append(q.get("depth_chart_dt", pd.NaT))
            else:
                ids.append(np.nan)
                names.append(np.nan)
                sources.append("unverified_fallback")
                verified.append(False)
                dts.append(pd.NaT)

        d[id_col] = ids
        d[name_col] = names
        d[source_col] = sources
        d[verified_col] = verified
        d[dt_col] = dts

    return d


def infer_kickoff_utc(row) -> pd.Timestamp | pd.NaT:
    # nflverse schedule convention uses Eastern gametime.
    if pd.notna(row.get("gameday")) and pd.notna(row.get("gametime")):
        try:
            naive = pd.Timestamp(
                f"{row['gameday']} {row['gametime']}"
            )
            return (
                naive
                .tz_localize(ZoneInfo("America/New_York"))
                .tz_convert("UTC")
            )
        except Exception:
            pass

    for c in ["kickoff", "game_datetime", "start_time"]:
        if c in row and pd.notna(row[c]):
            try:
                return pd.to_datetime(row[c], utc=True)
            except Exception:
                pass

    return pd.NaT


def add_schedule_metadata(
    market: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    d = market.copy()

    wanted = [
        "game_id", "home_team", "away_team",
        "gameday", "gametime",
        "roof", "surface", "temp", "wind",
        "home_rest", "away_rest", "div_game",
    ]
    cols = [c for c in wanted if c in schedule.columns]

    sched = schedule[cols].copy()
    d = d.merge(
        sched,
        on=[
            c for c in ["game_id", "home_team", "away_team"]
            if c in sched.columns
        ],
        how="left",
        suffixes=("", "_schedule"),
    )

    for c in [
        "roof", "surface", "temp", "wind",
        "home_rest", "away_rest", "div_game",
    ]:
        sc = f"{c}_schedule"
        if sc in d.columns:
            if c in market.columns:
                d[c] = d[c].combine_first(d[sc])
            else:
                d[c] = d[sc]
            d = d.drop(columns=[sc])

    d["kickoff_utc"] = [
        infer_kickoff_utc(r)
        for _, r in d.iterrows()
    ]
    return d


def roof_classification(row):
    home = normalize_team(row["home_team"])

    if pd.notna(row.get("roof")):
        roof = str(row["roof"]).strip().lower()
        if any(x in roof for x in ["dome", "closed", "indoors"]):
            return "indoor", roof, True
        if any(x in roof for x in ["outdoor", "open"]):
            return "outdoor", roof, True

    # Manual explicit roof status, if supplied.
    if pd.notna(row.get("roof_override")):
        roof = str(row["roof_override"]).strip().lower()
        if any(x in roof for x in ["dome", "closed", "indoors"]):
            return "indoor", roof, True
        if any(x in roof for x in ["outdoor", "open"]):
            return "outdoor", roof, True

    if home in FIXED_ROOF:
        return "indoor", FIXED_ROOF[home], True

    if home in RETRACTABLE_ROOF:
        return "unknown_retractable", "retractable", False

    if home in OUTDOOR_ROOF:
        return "outdoor", "outdoors", True

    return "unknown", np.nan, False


def open_meteo_weather(
    lat: float,
    lon: float,
    kickoff_utc: pd.Timestamp,
):
    if pd.isna(kickoff_utc):
        return None

    kickoff_utc = pd.Timestamp(kickoff_utc).tz_convert("UTC")
    date = kickoff_utc.strftime("%Y-%m-%d")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
        "start_date": date,
        "end_date": date,
    }

    url = OPEN_METEO_URL + "?" + urllib.parse.urlencode(params)
    data = read_json_url(url)

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    winds = hourly.get("wind_speed_10m", [])

    if not times or not temps or not winds:
        return None

    t = pd.to_datetime(times, utc=True)
    delta = np.abs((t - kickoff_utc).total_seconds())
    idx = int(np.argmin(delta))

    return {
        "temp": float(temps[idx]),
        "wind": float(winds[idx]),
        "weather_time_utc": str(t[idx]),
        "weather_source": "open_meteo",
    }


def add_weather_and_roof(market: pd.DataFrame) -> pd.DataFrame:
    d = market.copy()

    out_rows = []
    for _, r0 in d.iterrows():
        r = r0.copy()
        roof_type, resolved_roof, roof_known = roof_classification(r)

        r["roof_type_v30"] = roof_type
        r["roof_known_v30"] = roof_known

        if pd.isna(r.get("roof")) and pd.notna(resolved_roof):
            r["roof"] = resolved_roof

        if roof_type == "indoor":
            r["temp"] = 70.0
            r["wind"] = 0.0
            r["weather_source_v30"] = "indoor_neutral"
            r["weather_known_v30"] = True
            out_rows.append(r)
            continue

        if roof_type == "unknown_retractable":
            r["weather_source_v30"] = "roof_status_required"
            r["weather_known_v30"] = False
            out_rows.append(r)
            continue

        existing_temp = pd.to_numeric(
            pd.Series([r.get("temp")]), errors="coerce"
        ).iloc[0]
        existing_wind = pd.to_numeric(
            pd.Series([r.get("wind")]), errors="coerce"
        ).iloc[0]

        if pd.notna(existing_temp) and pd.notna(existing_wind):
            r["weather_source_v30"] = "market_or_schedule"
            r["weather_known_v30"] = True
            out_rows.append(r)
            continue

        lat = r.get("stadium_lat", np.nan)
        lon = r.get("stadium_lon", np.nan)

        if pd.isna(lat) or pd.isna(lon):
            coord = STADIUMS.get(normalize_team(r["home_team"]))
            if coord:
                lat, lon = coord

        if pd.isna(lat) or pd.isna(lon):
            r["weather_source_v30"] = "coordinates_missing"
            r["weather_known_v30"] = False
            out_rows.append(r)
            continue

        try:
            wx = open_meteo_weather(
                float(lat),
                float(lon),
                r.get("kickoff_utc"),
            )
        except Exception as e:
            print(
                f"WARNING: weather fetch failed for {r['game_id']}: {e}"
            )
            wx = None

        if wx:
            r["temp"] = wx["temp"]
            r["wind"] = wx["wind"]
            r["weather_time_utc"] = wx["weather_time_utc"]
            r["weather_source_v30"] = wx["weather_source"]
            r["weather_known_v30"] = True
        else:
            r["weather_source_v30"] = "forecast_unavailable"
            r["weather_known_v30"] = False

        out_rows.append(r)

    return pd.DataFrame(out_rows)


def refresh_odds_from_api(
    market: pd.DataFrame,
    bookmaker: str,
    api_key: str,
) -> pd.DataFrame:
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "bookmakers": bookmaker,
    }

    data = read_json_url(
        ODDS_API_URL + "?" + urllib.parse.urlencode(params)
    )

    d = market.copy()
    by_match = {}

    for event in data:
        home = normalize_team(
            FULL_NAME_TO_TEAM.get(event.get("home_team"))
        )
        away = normalize_team(
            FULL_NAME_TO_TEAM.get(event.get("away_team"))
        )

        if not home or not away:
            continue

        books = event.get("bookmakers", [])
        book = next(
            (b for b in books if b.get("key") == bookmaker),
            books[0] if books else None,
        )
        if not book:
            continue

        item = {
            "market_updated_at": book.get("last_update"),
            "market_source": bookmaker,
        }

        for m in book.get("markets", []):
            key = m.get("key")
            outcomes = m.get("outcomes", [])

            if key == "spreads":
                for o in outcomes:
                    team = FULL_NAME_TO_TEAM.get(o.get("name"))
                    team = normalize_team(team)
                    if team == home:
                        item["home_spread"] = o.get("point")
                        item["home_spread_odds"] = o.get("price")
                    elif team == away:
                        item["away_spread_odds"] = o.get("price")

            elif key == "totals":
                for o in outcomes:
                    if o.get("name") == "Over":
                        item["total_line"] = o.get("point")
                        item["over_odds"] = o.get("price")
                    elif o.get("name") == "Under":
                        item["under_odds"] = o.get("price")

        by_match[(away, home)] = item

    for idx, r in d.iterrows():
        item = by_match.get(
            (normalize_team(r["away_team"]), normalize_team(r["home_team"]))
        )
        if not item:
            continue
        for c, v in item.items():
            if v is not None:
                d.loc[idx, c] = v

    return d


def add_market_freshness(
    market: pd.DataFrame,
    max_age_minutes: float = 30.0,
) -> pd.DataFrame:
    d = market.copy()

    if "market_updated_at" not in d:
        d["market_updated_at"] = pd.NaT

    ts = pd.to_datetime(
        d["market_updated_at"],
        errors="coerce",
        utc=True,
    )

    now = pd.Timestamp.now(tz="UTC")
    age = (now - ts).dt.total_seconds() / 60.0

    d["market_age_minutes"] = age
    d["market_fresh_v30"] = (
        ts.notna()
        & (age >= -5)
        & (age <= float(max_age_minutes))
    )

    if "market_source" not in d:
        d["market_source"] = "manual_csv"

    return d


def add_readiness(market: pd.DataFrame) -> pd.DataFrame:
    d = market.copy()

    qb_ok = (
        d["home_qb_verified"].fillna(False)
        & d["away_qb_verified"].fillna(False)
    )

    market_ok = d["market_fresh_v30"].fillna(False)

    d["spread_input_ready_v30"] = market_ok & qb_ok

    surface_known = (
        d["surface"].notna()
        if "surface" in d.columns
        else pd.Series(False, index=d.index)
    )

    d["total_input_ready_v30"] = (
        d["spread_input_ready_v30"]
        & d["weather_known_v30"].fillna(False)
        & d["roof_known_v30"].fillna(False)
        & surface_known
    )

    reasons = []
    for _, r in d.iterrows():
        bad = []
        if not bool(r["market_fresh_v30"]):
            bad.append("stale_or_untimestamped_market")
        if not bool(r["home_qb_verified"]):
            bad.append("home_qb_unverified")
        if not bool(r["away_qb_verified"]):
            bad.append("away_qb_unverified")
        if not bool(r["roof_known_v30"]):
            bad.append("roof_unresolved")
        if not bool(r["weather_known_v30"]):
            bad.append("weather_unavailable")
        if not surface_known.loc[r.name]:
            bad.append("surface_unknown")
        reasons.append("|".join(bad))

    d["input_warnings_v30"] = reasons
    return d


def harden_market(
    market: pd.DataFrame,
    season: int,
    week: int,
    max_age_minutes: float = 30.0,
    refresh_odds: bool = False,
    bookmaker: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    d = normalize_market_teams(market)

    if refresh_odds:
        if not bookmaker:
            raise ValueError("--bookmaker is required with --refresh-odds.")
        if not api_key:
            raise ValueError(
                "THE_ODDS_API_KEY is required with --refresh-odds."
            )
        d = refresh_odds_from_api(d, bookmaker, api_key)

    schedule = load_schedule(season, week)
    d = add_schedule_metadata(d, schedule)

    depth = load_depth_charts(season)
    qbs = parse_depth_chart_qbs(depth)
    d = add_qb_inputs(d, qbs)

    d = add_weather_and_roof(d)
    d = add_market_freshness(
        d,
        max_age_minutes=max_age_minutes,
    )
    d = add_readiness(d)

    return d


def print_audit(d: pd.DataFrame):
    cols = [
        "game_id",
        "away_team",
        "home_team",
        "market_source",
        "market_age_minutes",
        "market_fresh_v30",
        "away_qb_name",
        "away_qb_input_source",
        "home_qb_name",
        "home_qb_input_source",
        "roof",
        "roof_type_v30",
        "temp",
        "wind",
        "weather_source_v30",
        "spread_input_ready_v30",
        "total_input_ready_v30",
        "input_warnings_v30",
    ]

    cols = [c for c in cols if c in d.columns]
    print("\nV3.0 LIVE INPUT AUDIT")
    print(d[cols].to_string(index=False))

    print("\nREADY COUNTS")
    print(
        "Spread-ready:",
        int(d["spread_input_ready_v30"].sum()),
        "/",
        len(d),
    )
    print(
        "Total-ready:",
        int(d["total_input_ready_v30"].sum()),
        "/",
        len(d),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--market-csv",
        default="live_data/week3_2026_market.csv",
    )
    ap.add_argument(
        "--output",
        default="live_data/week3_2026_market_hardened_v30.csv",
    )
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=3)
    ap.add_argument("--market-max-age-minutes", type=float, default=30.0)
    ap.add_argument("--refresh-odds", action="store_true")
    ap.add_argument("--bookmaker", default=None)
    args = ap.parse_args()

    market = pd.read_csv(args.market_csv)

    d = harden_market(
        market,
        season=args.season,
        week=args.week,
        max_age_minutes=args.market_max_age_minutes,
        refresh_odds=args.refresh_odds,
        bookmaker=args.bookmaker,
        api_key=os.environ.get("THE_ODDS_API_KEY"),
    )

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(p, index=False)

    print_audit(d)
    print(f"\nSaved hardened market: {p}")


if __name__ == "__main__":
    main()
