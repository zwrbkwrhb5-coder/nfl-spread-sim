from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (
    BASE_COLS,
    ADJ_COLS,
    build_team_games,
    build_features,
)

from .qb_layer_v17 import (
    build_qb_game_table,
    add_pregame_qb_form,
)

from .qb_safe_v28 import QB_METRICS
from .context_weather_v24 import normalize_context


NFLVERSE_PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp/play_by_play_{season}.parquet"
)

NFLVERSE_GAMES_URL = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)


SPREAD_FEATURES = [
    "diff_off_epa",
    "diff_def_epa",
    "diff_off_success",
    "diff_def_success",
] + [f"diff_{m}_v28" for m in QB_METRICS]


TOTAL_FEATURES = [
    "sum_off_epa",
    "sum_def_epa",
    "sum_off_success",
    "sum_def_success",
    "sum_points_for",
    "sum_points_against",
    "home_pre_adj_off_epa",
    "away_pre_adj_off_epa",
    "home_pre_adj_def_epa",
    "away_pre_adj_def_epa",
    "home_pre_explosive_pass_rate",
    "away_pre_explosive_pass_rate",
    "home_pre_turnover_rate",
    "away_pre_turnover_rate",
] + [
    f"{side}_pre_{m}_v28"
    for side in ["home", "away"]
    for m in QB_METRICS
] + [
    "divisional_game",
    "indoor",
    "grass",
    "cold_degrees",
    "wind_over_10",
    "cold_game",
    "windy_game",
]


def american_break_even(odds):
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


def load_current_pbp(season: int) -> pd.DataFrame:
    print(f"Loading {season} PBP...")
    return pd.read_parquet(NFLVERSE_PBP_URL.format(season=season))


def _ewm_last(series, halflife):
    s = pd.to_numeric(series, errors="coerce")
    x = s.ewm(halflife=halflife, adjust=False).mean()
    return float(x.iloc[-1]) if len(x) and pd.notna(x.iloc[-1]) else np.nan


def build_consistent_team_state(
    historical_team_games_path: str,
    current_pbp: pd.DataFrame,
    target_season: int,
    target_week: int,
    halflife: float = 6.0,
    min_games: int = 3,
) -> pd.DataFrame:
    """
    Uses the SAME raw -> pregame -> opponent-adjusted feature logic as
    src.features used to build the historical training table.

    Historical `team_games.parquet` supplies all seasons through 2025.
    Current-season PBP supplies completed 2026 games.
    """
    hp = Path(historical_team_games_path)
    if not hp.exists():
        raise FileNotFoundError(
            f"{hp} not found. v2.8 refuses to substitute raw EPA for "
            "opponent-adjusted live features. Rebuild the historical "
            "team_games artifact first."
        )

    hist = pd.read_parquet(hp)

    current = current_pbp[
        (pd.to_numeric(current_pbp["season"], errors="coerce") < target_season)
        | (
            (pd.to_numeric(current_pbp["season"], errors="coerce") == target_season)
            & (pd.to_numeric(current_pbp["week"], errors="coerce") < target_week)
        )
    ].copy()

    current_tg = build_team_games(current) if len(current) else pd.DataFrame()

    raw_cols = [
        "game_id", "season", "week", "team", "opponent", "is_home",
        "home_score", "away_score", "points_for", "points_against",
        *BASE_COLS,
    ]
    if "plays" in hist.columns:
        raw_cols.append("plays")

    hist_raw = hist[
        (pd.to_numeric(hist["season"], errors="coerce") < target_season)
    ].copy()

    for c in raw_cols:
        if c not in hist_raw.columns:
            hist_raw[c] = np.nan
        if not current_tg.empty and c not in current_tg.columns:
            current_tg[c] = np.nan

    combined = pd.concat(
        [
            hist_raw[raw_cols],
            current_tg[raw_cols] if not current_tg.empty else hist_raw.iloc[0:0][raw_cols],
        ],
        ignore_index=True,
    )

    combined = (
        combined
        .drop_duplicates(["game_id", "team"], keep="last")
        .sort_values(["season", "week", "game_id", "team"])
    )

    feat = build_features(
        combined,
        halflife_games=halflife,
        min_games=min_games,
    )

    feat = feat[
        (pd.to_numeric(feat["season"], errors="coerce") < target_season)
        | (
            (pd.to_numeric(feat["season"], errors="coerce") == target_season)
            & (pd.to_numeric(feat["week"], errors="coerce") < target_week)
        )
    ].copy()

    rows = []
    for team, g in feat.groupby("team"):
        g = g.sort_values(["season", "week", "game_id"])
        if g.empty:
            continue

        row = {"team": team}
        for c in BASE_COLS:
            row[f"pre_{c}"] = _ewm_last(g[c], halflife)
        for c in ADJ_COLS:
            row[f"pre_{c}"] = _ewm_last(g[c], halflife)

        rows.append(row)

    return pd.DataFrame(rows)


def _clean_qb_raw_history(qb_games: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "game_id", "season", "week", "team", "qb_id", "qb_name", "dropbacks",
        *QB_METRICS,
    ]
    d = qb_games.copy()
    for c in cols:
        if c not in d.columns:
            raise ValueError(f"QB history missing required column: {c}")
    return d[cols].copy()


def _resolve_override(g: pd.DataFrame, qb_id=None, qb_name=None):
    if qb_id is not None and str(qb_id).strip() and str(qb_id).lower() != "nan":
        m = g[g["qb_id"].astype(str) == str(qb_id)]
        if not m.empty:
            return str(qb_id)

    if qb_name is not None and str(qb_name).strip() and str(qb_name).lower() != "nan":
        want = str(qb_name).strip().lower()
        m = g[g["qb_name"].astype(str).str.strip().str.lower() == want]
        if not m.empty:
            return str(m.iloc[-1]["qb_id"])

    return None


def build_consistent_qb_state(
    historical_qb_games_csv: str,
    current_pbp: pd.DataFrame,
    market: pd.DataFrame,
    target_season: int,
    target_week: int,
    shrinkage_dropbacks: float = 100.0,
    halflife: float = 5.0,
) -> pd.DataFrame:
    """
    Historical and live QB values use the exact same transformation:
      raw QB game metrics
      -> EWM form
      -> prior-only pooled league target
      -> n/(n+100) shrinkage
    """
    hist = _clean_qb_raw_history(pd.read_csv(historical_qb_games_csv))

    cur_pbp = current_pbp[
        (pd.to_numeric(current_pbp["season"], errors="coerce") < target_season)
        | (
            (pd.to_numeric(current_pbp["season"], errors="coerce") == target_season)
            & (pd.to_numeric(current_pbp["week"], errors="coerce") < target_week)
        )
    ].copy()

    if len(cur_pbp):
        current_qg = _clean_qb_raw_history(build_qb_game_table(cur_pbp))
    else:
        current_qg = hist.iloc[0:0].copy()

    hist = hist[pd.to_numeric(hist["season"], errors="coerce") < target_season]

    qg = pd.concat([hist, current_qg], ignore_index=True)
    qg = (
        qg.drop_duplicates(["game_id", "team"], keep="last")
        .sort_values(["season", "week", "game_id", "team"])
        .reset_index(drop=True)
    )

    qform = add_pregame_qb_form(
        qg,
        halflife_games=halflife,
        min_games=2,
    )

    prior_form = qform[
        (pd.to_numeric(qform["season"], errors="coerce") < target_season)
        | (
            (pd.to_numeric(qform["season"], errors="coerce") == target_season)
            & (pd.to_numeric(qform["week"], errors="coerce") < target_week)
        )
    ].copy()

    targets = {}
    for m in QB_METRICS:
        x = pd.to_numeric(prior_form[f"pre_{m}"], errors="coerce").dropna()
        targets[m] = float(x.mean()) if len(x) else 0.0

    overrides = {}
    for _, r in market.iterrows():
        for side in ["home", "away"]:
            team = r[f"{side}_team"]
            overrides[team] = {
                "qb_id": r.get(f"{side}_qb_id", np.nan),
                "qb_name": r.get(f"{side}_qb_name", np.nan),
            }

    rows = []
    for team in sorted(set(market["home_team"]) | set(market["away_team"])):
        gt = qg[qg["team"] == team].sort_values(["season", "week", "game_id"])
        if gt.empty:
            continue

        ov = overrides.get(team, {})
        starter = _resolve_override(
            gt,
            qb_id=ov.get("qb_id"),
            qb_name=ov.get("qb_name"),
        )

        if starter is None:
            starter = str(gt.iloc[-1]["qb_id"])

        q = qg[qg["qb_id"].astype(str) == starter].sort_values(
            ["season", "week", "game_id"]
        )
        if q.empty:
            continue

        prior_db = float(pd.to_numeric(q["dropbacks"], errors="coerce").fillna(0).sum())
        w = prior_db / (prior_db + float(shrinkage_dropbacks))

        row = {
            "team": team,
            "qb_id": starter,
            "qb_name": str(q.iloc[-1]["qb_name"]),
            "qb_prior_dropbacks": prior_db,
            "qb_starter_source": (
                "market_override"
                if _resolve_override(
                    gt,
                    qb_id=ov.get("qb_id"),
                    qb_name=ov.get("qb_name"),
                ) is not None
                else "most_recent_team_starter"
            ),
        }

        for m in QB_METRICS:
            raw = _ewm_last(q[m], halflife)
            target = targets[m]
            row[f"pre_{m}_raw"] = raw
            row[f"pre_{m}_v28"] = target + w * (raw - target)
            row[f"qb_target_{m}_v28"] = target

        rows.append(row)

    return pd.DataFrame(rows)


def build_live_context(market: pd.DataFrame, season: int, week: int):
    games = pd.read_csv(NFLVERSE_GAMES_URL)
    g = games[
        (pd.to_numeric(games["season"], errors="coerce") == season)
        & (pd.to_numeric(games["week"], errors="coerce") == week)
    ].copy()

    wanted = [
        "game_id", "season", "week", "home_team", "away_team",
        "home_rest", "away_rest", "div_game", "roof", "surface", "temp", "wind",
    ]
    for c in wanted:
        if c not in g.columns:
            g[c] = np.nan

    merged = market[
        ["game_id", "season", "week", "home_team", "away_team"]
    ].merge(
        g[wanted],
        on=["game_id", "season", "week", "home_team", "away_team"],
        how="left",
    )

    # Optional market-file overrides take priority.
    for c in ["home_rest", "away_rest", "div_game", "roof", "surface", "temp", "wind"]:
        if c in market.columns:
            ov = market[["game_id", c]].rename(columns={c: f"{c}_override"})
            merged = merged.merge(ov, on="game_id", how="left")
            merged[c] = merged[f"{c}_override"].combine_first(merged[c])
            merged = merged.drop(columns=[f"{c}_override"])

    context = normalize_context(merged)

    outdoor = context["outdoor"].eq(1)
    weather_known = (
        pd.to_numeric(merged["temp"], errors="coerce").notna()
        & pd.to_numeric(merged["wind"], errors="coerce").notna()
    )
    context["weather_known_v28"] = (~outdoor) | weather_known

    return context


def build_live_features(market, team_state, qb_state, context):
    h = team_state.add_prefix("home_").rename(columns={"home_team": "home_team"})
    a = team_state.add_prefix("away_").rename(columns={"away_team": "away_team"})

    hq = qb_state.add_prefix("home_").rename(columns={"home_team": "home_team"})
    aq = qb_state.add_prefix("away_").rename(columns={"away_team": "away_team"})

    x = (
        market
        .merge(h, on="home_team", how="left")
        .merge(a, on="away_team", how="left")
        .merge(hq, on="home_team", how="left")
        .merge(aq, on="away_team", how="left")
        .merge(
            context.drop(
                columns=["season", "week", "home_team", "away_team"],
                errors="ignore",
            ),
            on="game_id",
            how="left",
        )
    )

    for c in [
        "off_epa", "def_epa", "off_success", "def_success",
    ]:
        x[f"diff_{c}"] = x[f"home_pre_{c}"] - x[f"away_pre_{c}"]

    for c in [
        "off_epa", "def_epa", "off_success", "def_success",
        "points_for", "points_against",
    ]:
        x[f"sum_{c}"] = x[f"home_pre_{c}"] + x[f"away_pre_{c}"]

    for m in QB_METRICS:
        x[f"diff_{m}_v28"] = (
            x[f"home_pre_{m}_v28"] - x[f"away_pre_{m}_v28"]
        )

    return x


def empirical_prob(errors, threshold):
    e = np.asarray(errors, dtype=float)
    e = e[np.isfinite(e)]
    return float(np.mean(e > threshold)) if len(e) else np.nan


def fit_live_isotonic(calibrated_sides: pd.DataFrame, market_name: str):
    d = calibrated_sides[
        (calibrated_sides["market"] == market_name)
        & calibrated_sides["win"].notna()
        & calibrated_sides["raw_probability"].notna()
    ].copy()

    if len(d) < 100 or d["win"].nunique() < 2:
        return None

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(
        d["raw_probability"].to_numpy(float),
        d["win"].to_numpy(float),
    )
    return iso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--training-csv",
        default="artifacts/dual_market/training_games_qb_safe_context_v28.csv",
    )
    ap.add_argument(
        "--oos-csv",
        default="artifacts/dual_market/oos_qb_safe_context_v28.csv",
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
        "--market-csv",
        default="live_data/week3_2026_market.csv",
    )
    ap.add_argument(
        "--calibrated-sides",
        default="artifacts/calibration_v28/calibrated_sides.csv",
    )
    ap.add_argument(
        "--total-params",
        default="artifacts/calibration_v28/latest_params.json",
    )
    ap.add_argument(
        "--spread-params",
        default="artifacts/spread_stability_v28/latest_params.json",
    )
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=3)
    ap.add_argument("--shrinkage-dropbacks", type=float, default=100.0)
    ap.add_argument(
        "--output",
        default="artifacts/live/week3_2026_ranked_v28.csv",
    )
    args = ap.parse_args()

    train = pd.read_csv(args.training_csv)
    market = pd.read_csv(args.market_csv)

    current_pbp = load_current_pbp(args.season)

    team_state = build_consistent_team_state(
        args.team_games,
        current_pbp,
        args.season,
        args.week,
    )

    qb_state = build_consistent_qb_state(
        args.qb_games,
        current_pbp,
        market,
        args.season,
        args.week,
        shrinkage_dropbacks=args.shrinkage_dropbacks,
    )

    context = build_live_context(
        market,
        args.season,
        args.week,
    )

    live = build_live_features(
        market,
        team_state,
        qb_state,
        context,
    )

    # Strictly refuse totals where future outdoor weather is unknown.
    if "weather_known_v28" in live.columns:
        unknown = live[
            (~live["weather_known_v28"].fillna(False))
        ][["game_id", "away_team", "home_team"]]
        if len(unknown):
            print(
                "\nWARNING: outdoor weather missing for these games. "
                "v2.8 will still produce spread projections but will not "
                "rank totals until temp/wind are supplied in the market CSV:"
            )
            print(unknown.to_string(index=False))

    tr_s = train.dropna(subset=SPREAD_FEATURES + ["actual_margin"])
    tr_t = train.dropna(subset=TOTAL_FEATURES + ["actual_total"])

    sm = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    tm = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])

    sm.fit(tr_s[SPREAD_FEATURES], tr_s["actual_margin"])
    tm.fit(tr_t[TOTAL_FEATURES], tr_t["actual_total"])

    live["spread_ready_v28"] = ~live[SPREAD_FEATURES].isna().any(axis=1)
    live["total_ready_v28"] = (
        ~live[TOTAL_FEATURES].isna().any(axis=1)
        & live.get("weather_known_v28", True)
    )

    live["model_margin"] = np.nan
    live["model_total"] = np.nan

    ms = live["spread_ready_v28"]
    mt = live["total_ready_v28"]

    live.loc[ms, "model_margin"] = sm.predict(live.loc[ms, SPREAD_FEATURES])
    live.loc[mt, "model_total"] = tm.predict(live.loc[mt, TOTAL_FEATURES])

    oos = pd.read_csv(args.oos_csv)
    mres = (
        pd.to_numeric(oos["actual_margin"], errors="coerce")
        - pd.to_numeric(oos["pred_margin"], errors="coerce")
    ).dropna().to_numpy(float)

    tres = (
        pd.to_numeric(oos["actual_total"], errors="coerce")
        - pd.to_numeric(oos["pred_total"], errors="coerce")
    ).dropna().to_numpy(float)

    calibration = (
        pd.read_csv(args.calibrated_sides)
        if Path(args.calibrated_sides).exists()
        else None
    )

    spread_iso = fit_live_isotonic(calibration, "spread") if calibration is not None else None
    total_iso = fit_live_isotonic(calibration, "total") if calibration is not None else None

    spread_params = {
        "probability_shrinkage_lambda": 0.25,
        "minimum_edge": 0.02,
    }
    total_params = {
        "probability_shrinkage_lambda": 0.50,
        "minimum_edge": 0.03,
    }

    if Path(args.spread_params).exists():
        spread_params.update(
            json.loads(Path(args.spread_params).read_text())["spread"]
        )
    if Path(args.total_params).exists():
        total_params.update(
            json.loads(Path(args.total_params).read_text())["total"]
        )

    rows = []

    for _, r in live.iterrows():
        if bool(r["spread_ready_v28"]):
            hthr = -(float(r["model_margin"]) + float(r["home_spread"]))
            p_home = empirical_prob(mres, hthr)
            p_away = 1.0 - p_home

            for side, team, p, odds in [
                ("home", r["home_team"], p_home, r["home_spread_odds"]),
                ("away", r["away_team"], p_away, r["away_spread_odds"]),
            ]:
                p_iso = float(spread_iso.predict([p])[0]) if spread_iso is not None else p
                lam = float(spread_params["probability_shrinkage_lambda"])
                p_sel = 0.5 + lam * (p_iso - 0.5)
                be = american_break_even(odds)
                rows.append({
                    "game_id": r["game_id"],
                    "market": "spread",
                    "side": side,
                    "pick": team,
                    "odds": float(odds),
                    "raw_probability": p,
                    "calibrated_probability": p_iso,
                    "selected_probability": p_sel,
                    "break_even": be,
                    "edge": p_sel - be,
                    "minimum_edge": float(spread_params["minimum_edge"]),
                    "model_margin": float(r["model_margin"]),
                    "market_home_spread": float(r["home_spread"]),
                    "model_total": (
                        float(r["model_total"])
                        if pd.notna(r["model_total"])
                        else np.nan
                    ),
                    "market_total": float(r["total_line"]),
                })

        if bool(r["total_ready_v28"]):
            tthr = float(r["total_line"]) - float(r["model_total"])
            p_over = empirical_prob(tres, tthr)
            p_under = 1.0 - p_over

            for side, pick, p, odds in [
                ("over", "OVER", p_over, r["over_odds"]),
                ("under", "UNDER", p_under, r["under_odds"]),
            ]:
                p_iso = float(total_iso.predict([p])[0]) if total_iso is not None else p
                lam = float(total_params["probability_shrinkage_lambda"])
                p_sel = 0.5 + lam * (p_iso - 0.5)
                be = american_break_even(odds)
                rows.append({
                    "game_id": r["game_id"],
                    "market": "total",
                    "side": side,
                    "pick": pick,
                    "odds": float(odds),
                    "raw_probability": p,
                    "calibrated_probability": p_iso,
                    "selected_probability": p_sel,
                    "break_even": be,
                    "edge": p_sel - be,
                    "minimum_edge": float(total_params["minimum_edge"]),
                    "model_margin": (
                        float(r["model_margin"])
                        if pd.notna(r["model_margin"])
                        else np.nan
                    ),
                    "market_home_spread": float(r["home_spread"]),
                    "model_total": float(r["model_total"]),
                    "market_total": float(r["total_line"]),
                })

    ranked = pd.DataFrame(rows)

    if ranked.empty:
        raise RuntimeError("No live rows were rankable.")

    # One recommendation per game/market. This prevents both sides of the
    # same spread from appearing in Top 5.
    ranked = (
        ranked.sort_values(
            ["game_id", "market", "edge"],
            ascending=[True, True, False],
        )
        .drop_duplicates(["game_id", "market"])
        .copy()
    )

    ranked["qualifies"] = ranked["edge"] >= ranked["minimum_edge"]
    ranked = ranked.sort_values(
        ["qualifies", "edge"],
        ascending=[False, False],
    ).reset_index(drop=True)

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(p, index=False)

    print("\nV2.8 TOP QUALIFYING OPPORTUNITIES")
    cols = [
        "game_id", "market", "pick", "odds",
        "raw_probability", "calibrated_probability",
        "selected_probability", "break_even", "edge",
        "minimum_edge", "qualifies",
        "model_margin", "market_home_spread",
        "model_total", "market_total",
    ]
    print(ranked[cols].head(15).to_string(index=False))
    print(f"\nSaved {p}")


if __name__ == "__main__":
    main()
