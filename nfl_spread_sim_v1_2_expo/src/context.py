from __future__ import annotations
import math
import numpy as np
import pandas as pd

CONTEXT_FEATURES = [
    "rest_diff_days",
    "home_short_week",
    "away_short_week",
    "home_off_bye",
    "away_off_bye",
    "travel_miles_diff",
    "time_zone_shift_diff",
    "divisional_game",
    "surface_grass",
    "surface_turf",
    "roof_indoor",
    "roof_closed",
    "temperature_f",
    "wind_mph",
    "precipitation_in",
]

def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.7613
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return 2 * r * math.asin(math.sqrt(a))

def add_schedule_context(games: pd.DataFrame) -> pd.DataFrame:
    """
    Requires:
      game_date
      home_team
      away_team
      season/week/game_id

    Computes days since each team's previous game using only past schedule info.
    """
    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce")

    long = pd.concat([
        g[["game_id","season","week","game_date","home_team"]]
            .rename(columns={"home_team":"team"})
            .assign(side="home"),
        g[["game_id","season","week","game_date","away_team"]]
            .rename(columns={"away_team":"team"})
            .assign(side="away"),
    ], ignore_index=True)

    long = long.sort_values(["team","game_date","game_id"])
    long["prev_game_date"] = long.groupby("team")["game_date"].shift(1)
    long["rest_days"] = (long["game_date"] - long["prev_game_date"]).dt.days

    rest = long[["game_id","team","side","rest_days"]]
    home = rest[rest["side"].eq("home")][["game_id","rest_days"]].rename(
        columns={"rest_days":"home_rest_days"}
    )
    away = rest[rest["side"].eq("away")][["game_id","rest_days"]].rename(
        columns={"rest_days":"away_rest_days"}
    )

    g = g.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")
    g["rest_diff_days"] = g["home_rest_days"] - g["away_rest_days"]
    g["home_short_week"] = g["home_rest_days"].le(6).astype(float)
    g["away_short_week"] = g["away_rest_days"].le(6).astype(float)
    g["home_off_bye"] = g["home_rest_days"].ge(10).astype(float)
    g["away_off_bye"] = g["away_rest_days"].ge(10).astype(float)
    return g

def add_travel_context(
    games: pd.DataFrame,
    team_locations: pd.DataFrame,
    stadium_locations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds travel distance and time-zone shift for each team.

    team_locations columns:
      team, lat, lon, utc_offset_std

    stadium_locations columns:
      stadium, lat, lon, utc_offset_std
    """
    g = games.copy()
    teams = team_locations.set_index("team")
    stadiums = stadium_locations.set_index("stadium")

    h_miles, a_miles = [], []
    h_tz, a_tz = [], []

    for _, r in g.iterrows():
        if r["stadium"] not in stadiums.index:
            h_miles.append(np.nan); a_miles.append(np.nan)
            h_tz.append(np.nan); a_tz.append(np.nan)
            continue

        st = stadiums.loc[r["stadium"]]
        vals = []
        tzs = []
        for team in [r["home_team"], r["away_team"]]:
            if team not in teams.index:
                vals.append(np.nan); tzs.append(np.nan)
                continue
            t = teams.loc[team]
            vals.append(haversine_miles(t["lat"], t["lon"], st["lat"], st["lon"]))
            tzs.append(abs(float(t["utc_offset_std"]) - float(st["utc_offset_std"])))

        h_miles.append(vals[0]); a_miles.append(vals[1])
        h_tz.append(tzs[0]); a_tz.append(tzs[1])

    g["home_travel_miles"] = h_miles
    g["away_travel_miles"] = a_miles
    g["travel_miles_diff"] = g["home_travel_miles"] - g["away_travel_miles"]
    g["home_time_zone_shift"] = h_tz
    g["away_time_zone_shift"] = a_tz
    g["time_zone_shift_diff"] = g["home_time_zone_shift"] - g["away_time_zone_shift"]
    return g

def add_game_environment(games: pd.DataFrame, game_context: pd.DataFrame) -> pd.DataFrame:
    """
    Merge per-game weather/roof/surface/division context.

    Required columns in game_context:
      game_id, divisional_game, surface, roof,
      temperature_f, wind_mph, precipitation_in
    """
    g = games.merge(game_context, on="game_id", how="left")

    g["surface_grass"] = g["surface"].astype(str).str.lower().str.contains("grass").astype(float)
    g["surface_turf"] = g["surface"].astype(str).str.lower().str.contains("turf|artificial").astype(float)
    roof = g["roof"].astype(str).str.lower()
    g["roof_indoor"] = roof.str.contains("dome|indoor").astype(float)
    g["roof_closed"] = roof.str.contains("closed").astype(float)

    for c in ["temperature_f","wind_mph","precipitation_in","divisional_game"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")

    return g

def add_context_interactions(games: pd.DataFrame):
    """
    Add interpretable interactions the model may find useful.
    """
    g = games.copy()
    feats = []

    if "away_pre_qb_cpoe" in g.columns:
        g["wind_x_away_qb_cpoe"] = g["wind_mph"] * g["away_pre_qb_cpoe"]
        feats.append("wind_x_away_qb_cpoe")

    if "away_pre_qb_sack_rate" in g.columns:
        g["short_week_x_away_qb_sack"] = g["away_short_week"] * g["away_pre_qb_sack_rate"]
        feats.append("short_week_x_away_qb_sack")

    g["travel_x_tz_away"] = g["away_travel_miles"] * g["away_time_zone_shift"]
    feats.append("travel_x_tz_away")

    g["wind_x_outdoor"] = g["wind_mph"] * (1 - g["roof_indoor"].fillna(0))
    feats.append("wind_x_outdoor")

    return g, feats
