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
)

from .context_weather_v24 import normalize_context


NFLVERSE_PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp/play_by_play_{season}.parquet"
)

NFLVERSE_GAMES_URL = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)

QB_METRICS = [
    "qb_epa_per_dropback",
    "qb_success_rate",
    "qb_cpoe",
    "qb_sack_rate",
    "qb_interception_rate",
    "qb_explosive_pass_rate",
]

SPREAD_FEATURES = [
    "diff_off_epa",
    "diff_def_epa",
    "diff_off_success",
    "diff_def_success",
] + [f"diff_{m}" for m in QB_METRICS]

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
    f"{side}_pre_{m}"
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


def american_break_even(odds: float) -> float:
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


def load_current_pbp(season: int) -> pd.DataFrame:
    print(f"Loading {season} PBP...")
    return pd.read_parquet(
        NFLVERSE_PBP_URL.format(season=season)
    )


def ewm_last(series, halflife: float):
    s = pd.to_numeric(series, errors="coerce")
    x = s.ewm(halflife=halflife, adjust=False).mean()
    return (
        float(x.iloc[-1])
        if len(x) and pd.notna(x.iloc[-1])
        else np.nan
    )


def build_consistent_team_state(
    historical_team_games_path: str,
    current_pbp: pd.DataFrame,
    target_season: int,
    target_week: int,
    halflife: float = 6.0,
    min_games: int = 3,
) -> pd.DataFrame:
    """
    Reuses the exact raw -> EWM -> opponent-adjusted pipeline used by
    historical model construction.

    v2.9 intentionally refuses to replace adjusted features with raw EPA.
    """
    hp = Path(historical_team_games_path)
    if not hp.exists():
        raise FileNotFoundError(
            f"{hp} not found. Rebuild the historical team-games artifact "
            "before running v2.9."
        )

    hist = pd.read_parquet(hp)

    current = current_pbp[
        (pd.to_numeric(current_pbp["season"], errors="coerce") < target_season)
        | (
            (
                pd.to_numeric(
                    current_pbp["season"],
                    errors="coerce",
                )
                == target_season
            )
            & (
                pd.to_numeric(
                    current_pbp["week"],
                    errors="coerce",
                )
                < target_week
            )
        )
    ].copy()

    current_tg = (
        build_team_games(current)
        if len(current)
        else pd.DataFrame()
    )

    raw_cols = [
        "game_id",
        "season",
        "week",
        "team",
        "opponent",
        "is_home",
        "home_score",
        "away_score",
        "points_for",
        "points_against",
        *BASE_COLS,
    ]

    if "plays" in hist.columns:
        raw_cols.append("plays")

    # Remove duplicate column names. points_for / points_against
    # are already part of BASE_COLS.
    raw_cols = list(dict.fromkeys(raw_cols))

    hist_raw = hist[
        pd.to_numeric(hist["season"], errors="coerce")
        < target_season
    ].copy()

    for c in raw_cols:
        if c not in hist_raw.columns:
            hist_raw[c] = np.nan
        if not current_tg.empty and c not in current_tg.columns:
            current_tg[c] = np.nan

    if current_tg.empty:
        combined = hist_raw[raw_cols].copy()
    else:
        combined = pd.concat(
            [hist_raw[raw_cols], current_tg[raw_cols]],
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
        (
            pd.to_numeric(feat["season"], errors="coerce")
            < target_season
        )
        | (
            (
                pd.to_numeric(
                    feat["season"],
                    errors="coerce",
                )
                == target_season
            )
            & (
                pd.to_numeric(
                    feat["week"],
                    errors="coerce",
                )
                < target_week
            )
        )
    ].copy()

    rows = []
    for team, g in feat.groupby("team"):
        g = g.sort_values(["season", "week", "game_id"])
        if g.empty:
            continue

        row = {"team": team}

        for c in BASE_COLS:
            row[f"pre_{c}"] = ewm_last(g[c], halflife)

        for c in ADJ_COLS:
            row[f"pre_{c}"] = ewm_last(g[c], halflife)

        rows.append(row)

    return pd.DataFrame(rows)


def clean_qb_games(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "game_id",
        "season",
        "week",
        "team",
        "qb_id",
        "qb_name",
        "dropbacks",
        *QB_METRICS,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"QB game table missing columns: {missing}"
        )
    return df[required].copy()


def resolve_qb_override(
    team_history: pd.DataFrame,
    qb_id=None,
    qb_name=None,
):
    if (
        qb_id is not None
        and str(qb_id).strip()
        and str(qb_id).lower() != "nan"
    ):
        m = team_history[
            team_history["qb_id"].astype(str)
            == str(qb_id)
        ]
        if not m.empty:
            return str(qb_id)

    if (
        qb_name is not None
        and str(qb_name).strip()
        and str(qb_name).lower() != "nan"
    ):
        want = str(qb_name).strip().lower()
        m = team_history[
            team_history["qb_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            == want
        ]
        if not m.empty:
            return str(m.iloc[-1]["qb_id"])

    return None


def build_raw_qb_state(
    historical_qb_games_csv: str,
    current_pbp: pd.DataFrame,
    market: pd.DataFrame,
    target_season: int,
    target_week: int,
    halflife: float = 5.0,
) -> pd.DataFrame:
    """
    Live raw QB form matching historical v1.7 semantics.

    Historical pregame feature at game t:
        EWM(QB games through t-1)

    Live upcoming-game feature:
        EWM(QB games through the most recent completed game)

    No league-average shrinkage is applied.
    """
    hp = Path(historical_qb_games_csv)
    if not hp.exists():
        raise FileNotFoundError(
            f"Missing QB history: {hp}"
        )

    hist = clean_qb_games(pd.read_csv(hp))
    hist = hist[
        pd.to_numeric(hist["season"], errors="coerce")
        < target_season
    ].copy()

    current = current_pbp[
        (
            pd.to_numeric(
                current_pbp["season"],
                errors="coerce",
            )
            < target_season
        )
        | (
            (
                pd.to_numeric(
                    current_pbp["season"],
                    errors="coerce",
                )
                == target_season
            )
            & (
                pd.to_numeric(
                    current_pbp["week"],
                    errors="coerce",
                )
                < target_week
            )
        )
    ].copy()

    current_qg = (
        clean_qb_games(build_qb_game_table(current))
        if len(current)
        else hist.iloc[0:0].copy()
    )

    qg = pd.concat(
        [hist, current_qg],
        ignore_index=True,
    )

    qg = (
        qg
        .drop_duplicates(["game_id", "team"], keep="last")
        .sort_values(["season", "week", "game_id", "team"])
        .reset_index(drop=True)
    )

    override_by_team = {}
    for _, r in market.iterrows():
        for side in ["home", "away"]:
            team = r[f"{side}_team"]
            override_by_team[team] = {
                "qb_id": r.get(f"{side}_qb_id", np.nan),
                "qb_name": r.get(
                    f"{side}_qb_name",
                    np.nan,
                ),
            }

    rows = []

    teams = sorted(
        set(market["home_team"])
        | set(market["away_team"])
    )

    for team in teams:
        tg = qg[qg["team"] == team].copy()
        if tg.empty:
            continue

        tg = tg.sort_values(
            ["season", "week", "game_id"]
        )

        override = override_by_team.get(team, {})
        resolved = resolve_qb_override(
            tg,
            qb_id=override.get("qb_id"),
            qb_name=override.get("qb_name"),
        )

        if resolved is None:
            starter = str(tg.iloc[-1]["qb_id"])
            starter_source = "most_recent_team_starter"
        else:
            starter = resolved
            starter_source = "market_override"

        q = qg[
            qg["qb_id"].astype(str) == starter
        ].sort_values(
            ["season", "week", "game_id"]
        )

        if q.empty:
            continue

        row = {
            "team": team,
            "qb_id": starter,
            "qb_name": str(q.iloc[-1]["qb_name"]),
            "qb_prior_dropbacks": float(
                pd.to_numeric(
                    q["dropbacks"],
                    errors="coerce",
                )
                .fillna(0.0)
                .sum()
            ),
            "qb_starter_source": starter_source,
        }

        for metric in QB_METRICS:
            row[f"pre_{metric}"] = ewm_last(
                q[metric],
                halflife,
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_live_context(
    market: pd.DataFrame,
    season: int,
    week: int,
):
    games = pd.read_csv(NFLVERSE_GAMES_URL)

    g = games[
        (
            pd.to_numeric(
                games["season"],
                errors="coerce",
            )
            == season
        )
        & (
            pd.to_numeric(
                games["week"],
                errors="coerce",
            )
            == week
        )
    ].copy()

    wanted = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_rest",
        "away_rest",
        "div_game",
        "roof",
        "surface",
        "temp",
        "wind",
    ]

    for c in wanted:
        if c not in g.columns:
            g[c] = np.nan

    merged = market[
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
        ]
    ].merge(
        g[wanted],
        on=[
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
        ],
        how="left",
    )

    # Manual current-game information in market CSV wins over schedule file.
    for c in [
        "home_rest",
        "away_rest",
        "div_game",
        "roof",
        "surface",
        "temp",
        "wind",
    ]:
        if c in market.columns:
            override = market[
                ["game_id", c]
            ].rename(
                columns={c: f"{c}_override"}
            )
            merged = merged.merge(
                override,
                on="game_id",
                how="left",
            )
            merged[c] = (
                merged[f"{c}_override"]
                .combine_first(merged[c])
            )
            merged = merged.drop(
                columns=[f"{c}_override"]
            )

    context = normalize_context(merged)

    outdoor = context["outdoor"].eq(1)

    weather_known = (
        pd.to_numeric(
            merged["temp"],
            errors="coerce",
        ).notna()
        & pd.to_numeric(
            merged["wind"],
            errors="coerce",
        ).notna()
    )

    context["weather_known_v29"] = (
        (~outdoor) | weather_known
    )

    return context


def build_live_features(
    market: pd.DataFrame,
    team_state: pd.DataFrame,
    qb_state: pd.DataFrame,
    context: pd.DataFrame,
):
    h = (
        team_state
        .add_prefix("home_")
        .rename(
            columns={"home_team": "home_team"}
        )
    )

    a = (
        team_state
        .add_prefix("away_")
        .rename(
            columns={"away_team": "away_team"}
        )
    )

    hq = (
        qb_state
        .add_prefix("home_")
        .rename(
            columns={"home_team": "home_team"}
        )
    )

    aq = (
        qb_state
        .add_prefix("away_")
        .rename(
            columns={"away_team": "away_team"}
        )
    )

    x = (
        market
        .merge(h, on="home_team", how="left")
        .merge(a, on="away_team", how="left")
        .merge(hq, on="home_team", how="left")
        .merge(aq, on="away_team", how="left")
        .merge(
            context.drop(
                columns=[
                    "season",
                    "week",
                    "home_team",
                    "away_team",
                ],
                errors="ignore",
            ),
            on="game_id",
            how="left",
        )
    )

    for c in [
        "off_epa",
        "def_epa",
        "off_success",
        "def_success",
    ]:
        x[f"diff_{c}"] = (
            x[f"home_pre_{c}"]
            - x[f"away_pre_{c}"]
        )

    for c in [
        "off_epa",
        "def_epa",
        "off_success",
        "def_success",
        "points_for",
        "points_against",
    ]:
        x[f"sum_{c}"] = (
            x[f"home_pre_{c}"]
            + x[f"away_pre_{c}"]
        )

    for metric in QB_METRICS:
        x[f"diff_{metric}"] = (
            x[f"home_pre_{metric}"]
            - x[f"away_pre_{metric}"]
        )

    return x


def empirical_probability(
    residuals,
    threshold: float,
):
    x = np.asarray(residuals, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.mean(x > threshold))


def fit_final_isotonic(
    calibrated_sides: pd.DataFrame,
    market_name: str,
):
    d = calibrated_sides[
        (calibrated_sides["market"] == market_name)
        & calibrated_sides["win"].notna()
        & calibrated_sides["raw_probability"].notna()
    ].copy()

    if len(d) < 100 or d["win"].nunique() < 2:
        raise RuntimeError(
            f"Not enough historical {market_name} "
            "calibration rows."
        )

    iso = IsotonicRegression(
        out_of_bounds="clip"
    )
    iso.fit(
        d["raw_probability"].to_numpy(float),
        d["win"].to_numpy(float),
    )
    return iso


def load_required_params(
    total_params_path: str,
    spread_params_path: str,
):
    tp = Path(total_params_path)
    sp = Path(spread_params_path)

    if not tp.exists():
        raise FileNotFoundError(
            f"Missing totals params: {tp}. "
            "Run latest_calibration_params_v26 first."
        )

    if not sp.exists():
        raise FileNotFoundError(
            f"Missing spread params: {sp}. "
            "Run latest_spread_stability_params_v27 first."
        )

    total = json.loads(
        tp.read_text(encoding="utf-8")
    )["total"]

    spread = json.loads(
        sp.read_text(encoding="utf-8")
    )["spread"]

    return total, spread


def validate_market_csv(market: pd.DataFrame):
    required = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_spread",
        "total_line",
        "home_spread_odds",
        "away_spread_odds",
        "over_odds",
        "under_odds",
    ]

    missing = [
        c for c in required
        if c not in market.columns
    ]

    if missing:
        raise ValueError(
            f"Market CSV missing columns: {missing}"
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--training-csv",
        default=(
            "artifacts/dual_market/"
            "training_games_qb_context_v24.csv"
        ),
    )

    ap.add_argument(
        "--oos-csv",
        default=(
            "artifacts/dual_market/"
            "oos_qb_raw_context_v28.csv"
        ),
    )

    ap.add_argument(
        "--team-games",
        default="artifacts/team_games.parquet",
    )

    ap.add_argument(
        "--qb-games",
        default=(
            "artifacts/dual_market/"
            "qb_games_v17.csv"
        ),
    )

    ap.add_argument(
        "--market-csv",
        default="live_data/week3_2026_market.csv",
    )

    ap.add_argument(
        "--calibrated-sides",
        default=(
            "artifacts/calibration_v28_raw/"
            "calibrated_sides.csv"
        ),
    )

    ap.add_argument(
        "--total-params",
        default=(
            "artifacts/calibration_v28_raw/"
            "latest_params.json"
        ),
    )

    ap.add_argument(
        "--spread-params",
        default=(
            "artifacts/spread_stability_v28_raw/"
            "latest_params.json"
        ),
    )

    ap.add_argument(
        "--season",
        type=int,
        default=2026,
    )

    ap.add_argument(
        "--week",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--output",
        default=(
            "artifacts/live/"
            "week3_2026_ranked_v29.csv"
        ),
    )

    args = ap.parse_args()

    train = pd.read_csv(args.training_csv)
    oos = pd.read_csv(args.oos_csv)
    market = pd.read_csv(args.market_csv)

    validate_market_csv(market)

    # Normalize market team aliases to nflverse team codes.
    TEAM_ALIASES = {
        "LAR": "LA",
    }

    market["home_team"] = market["home_team"].replace(TEAM_ALIASES)
    market["away_team"] = market["away_team"].replace(TEAM_ALIASES)

    # Keep game_id aligned with the normalized team codes as well.
    market["game_id"] = (
        market["game_id"]
        .astype(str)
        .str.replace("_LAR_", "_LA_", regex=False)
    )

    total_params, spread_params = (
        load_required_params(
            args.total_params,
            args.spread_params,
        )
    )

    calibration = pd.read_csv(
        args.calibrated_sides
    )

    current_pbp = load_current_pbp(
        args.season
    )

    team_state = build_consistent_team_state(
        args.team_games,
        current_pbp,
        args.season,
        args.week,
    )

    qb_state = build_raw_qb_state(
        args.qb_games,
        current_pbp,
        market,
        args.season,
        args.week,
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

    spread_train = train.dropna(
        subset=SPREAD_FEATURES
        + ["actual_margin"]
    )

    total_train = train.dropna(
        subset=TOTAL_FEATURES
        + ["actual_total"]
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

    live["spread_ready_v29"] = (
        ~live[SPREAD_FEATURES]
        .isna()
        .any(axis=1)
    )

    weather_ok = (
        live["weather_known_v29"]
        if "weather_known_v29" in live.columns
        else pd.Series(
            False,
            index=live.index,
        )
    )

    live["total_ready_v29"] = (
        ~live[TOTAL_FEATURES]
        .isna()
        .any(axis=1)
        & weather_ok
    )

    live["model_margin"] = np.nan
    live["model_total"] = np.nan

    smask = live["spread_ready_v29"]
    tmask = live["total_ready_v29"]

    live.loc[
        smask,
        "model_margin",
    ] = spread_model.predict(
        live.loc[
            smask,
            SPREAD_FEATURES,
        ]
    )

    live.loc[
        tmask,
        "model_total",
    ] = total_model.predict(
        live.loc[
            tmask,
            TOTAL_FEATURES,
        ]
    )

    missing_spread = live[
        ~live["spread_ready_v29"]
    ][
        [
            "game_id",
            "away_team",
            "home_team",
        ]
    ]

    missing_total = live[
        ~live["total_ready_v29"]
    ][
        [
            "game_id",
            "away_team",
            "home_team",
        ]
    ]

    if len(missing_spread):
        print(
            "\nSPREAD EXCLUSIONS — incomplete live features"
        )
        print(
            missing_spread.to_string(
                index=False
            )
        )

    if len(missing_total):
        print(
            "\nTOTAL EXCLUSIONS — incomplete features "
            "or outdoor weather unavailable"
        )
        print(
            missing_total.to_string(
                index=False
            )
        )

    margin_residuals = (
        pd.to_numeric(
            oos["actual_margin"],
            errors="coerce",
        )
        - pd.to_numeric(
            oos["pred_margin"],
            errors="coerce",
        )
    ).dropna().to_numpy(float)

    total_residuals = (
        pd.to_numeric(
            oos["actual_total"],
            errors="coerce",
        )
        - pd.to_numeric(
            oos["pred_total"],
            errors="coerce",
        )
    ).dropna().to_numpy(float)

    spread_iso = fit_final_isotonic(
        calibration,
        "spread",
    )

    total_iso = fit_final_isotonic(
        calibration,
        "total",
    )

    rows = []

    for _, r in live.iterrows():
        if bool(r["spread_ready_v29"]):
            threshold = -(
                float(r["model_margin"])
                + float(r["home_spread"])
            )

            p_home_raw = empirical_probability(
                margin_residuals,
                threshold,
            )

            p_away_raw = 1.0 - p_home_raw

            spread_options = [
                (
                    "home",
                    r["home_team"],
                    p_home_raw,
                    r["home_spread_odds"],
                ),
                (
                    "away",
                    r["away_team"],
                    p_away_raw,
                    r["away_spread_odds"],
                ),
            ]

            for side, pick, raw_p, odds in spread_options:
                calibrated_p = float(
                    spread_iso.predict(
                        [raw_p]
                    )[0]
                )

                lam = float(
                    spread_params[
                        "probability_shrinkage_lambda"
                    ]
                )

                selected_p = (
                    0.5
                    + lam
                    * (
                        calibrated_p
                        - 0.5
                    )
                )

                be = american_break_even(
                    odds
                )

                rows.append({
                    "game_id": r["game_id"],
                    "season": int(r["season"]),
                    "week": int(r["week"]),
                    "away_team": r["away_team"],
                    "home_team": r["home_team"],
                    "market": "spread",
                    "side": side,
                    "pick": pick,
                    "odds": float(odds),
                    "raw_probability": raw_p,
                    "calibrated_probability": calibrated_p,
                    "selected_probability": selected_p,
                    "break_even": be,
                    "edge": selected_p - be,
                    "minimum_edge": float(
                        spread_params[
                            "minimum_edge"
                        ]
                    ),
                    "model_margin": float(
                        r["model_margin"]
                    ),
                    "market_home_spread": float(
                        r["home_spread"]
                    ),
                    "model_total": (
                        float(
                            r["model_total"]
                        )
                        if pd.notna(
                            r["model_total"]
                        )
                        else np.nan
                    ),
                    "market_total": float(
                        r["total_line"]
                    ),
                    "home_qb": r.get(
                        "home_qb_name",
                        np.nan,
                    ),
                    "away_qb": r.get(
                        "away_qb_name",
                        np.nan,
                    ),
                    "home_qb_source": r.get(
                        "home_qb_starter_source",
                        np.nan,
                    ),
                    "away_qb_source": r.get(
                        "away_qb_starter_source",
                        np.nan,
                    ),
                })

        if bool(r["total_ready_v29"]):
            threshold = (
                float(r["total_line"])
                - float(r["model_total"])
            )

            p_over_raw = empirical_probability(
                total_residuals,
                threshold,
            )

            p_under_raw = (
                1.0 - p_over_raw
            )

            total_options = [
                (
                    "over",
                    "OVER",
                    p_over_raw,
                    r["over_odds"],
                ),
                (
                    "under",
                    "UNDER",
                    p_under_raw,
                    r["under_odds"],
                ),
            ]

            for side, pick, raw_p, odds in total_options:
                calibrated_p = float(
                    total_iso.predict(
                        [raw_p]
                    )[0]
                )

                lam = float(
                    total_params[
                        "probability_shrinkage_lambda"
                    ]
                )

                selected_p = (
                    0.5
                    + lam
                    * (
                        calibrated_p
                        - 0.5
                    )
                )

                be = american_break_even(
                    odds
                )

                rows.append({
                    "game_id": r["game_id"],
                    "season": int(r["season"]),
                    "week": int(r["week"]),
                    "away_team": r["away_team"],
                    "home_team": r["home_team"],
                    "market": "total",
                    "side": side,
                    "pick": pick,
                    "odds": float(odds),
                    "raw_probability": raw_p,
                    "calibrated_probability": calibrated_p,
                    "selected_probability": selected_p,
                    "break_even": be,
                    "edge": selected_p - be,
                    "minimum_edge": float(
                        total_params[
                            "minimum_edge"
                        ]
                    ),
                    "model_margin": (
                        float(
                            r["model_margin"]
                        )
                        if pd.notna(
                            r["model_margin"]
                        )
                        else np.nan
                    ),
                    "market_home_spread": float(
                        r["home_spread"]
                    ),
                    "model_total": float(
                        r["model_total"]
                    ),
                    "market_total": float(
                        r["total_line"]
                    ),
                    "home_qb": r.get(
                        "home_qb_name",
                        np.nan,
                    ),
                    "away_qb": r.get(
                        "away_qb_name",
                        np.nan,
                    ),
                    "home_qb_source": r.get(
                        "home_qb_starter_source",
                        np.nan,
                    ),
                    "away_qb_source": r.get(
                        "away_qb_starter_source",
                        np.nan,
                    ),
                })

    ranked = pd.DataFrame(rows)

    if ranked.empty:
        raise RuntimeError(
            "No live games had complete v2.9 features."
        )

    # Critical UI/model rule:
    # exactly one side per game/market.
    ranked = (
        ranked
        .sort_values(
            ["game_id", "market", "edge"],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .drop_duplicates(
            ["game_id", "market"]
        )
        .copy()
    )

    ranked["qualifies"] = (
        ranked["edge"]
        >= ranked["minimum_edge"]
    )

    ranked = ranked.sort_values(
        ["qualifies", "edge"],
        ascending=[False, False],
    ).reset_index(drop=True)

    out = Path(args.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    ranked.to_csv(
        out,
        index=False,
    )

    print("\nV2.9 MODEL CONFIG")
    print(
        json.dumps(
            {
                "spread": spread_params,
                "total": total_params,
                "raw_qb_shrinkage": False,
                "one_side_per_game_market": True,
                "true_opponent_adjustment_live": True,
            },
            indent=2,
        )
    )

    print("\nV2.9 TOP QUALIFYING OPPORTUNITIES")

    show = [
        "game_id",
        "market",
        "pick",
        "odds",
        "raw_probability",
        "calibrated_probability",
        "selected_probability",
        "break_even",
        "edge",
        "minimum_edge",
        "qualifies",
        "model_margin",
        "market_home_spread",
        "model_total",
        "market_total",
        "away_qb",
        "home_qb",
    ]

    print(
        ranked[show]
        .head(15)
        .to_string(index=False)
    )

    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
