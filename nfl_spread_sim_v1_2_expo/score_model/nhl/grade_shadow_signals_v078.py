from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import urllib.request

import numpy as np
import pandas as pd


LOG = Path(
    "artifacts/nhl/live/nhl_shadow_signals.csv"
)

SUMMARY = Path(
    "artifacts/nhl/live/nhl_shadow_summary.csv"
)

API_BASE = (
    "https://api-web.nhle.com/v1/score"
)


def clean_str(x):
    if pd.isna(x):
        return ""

    return str(x).strip()


def is_blank(x):
    return clean_str(x) == ""


def fetch_scoreboard(date_str):
    url = f"{API_BASE}/{date_str}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0 NHL-shadow-grader"
        },
    )

    with urllib.request.urlopen(
        req,
        timeout=20,
    ) as response:
        return json.load(response)


def find_game(
    scoreboard,
    away,
    home,
):
    away = away.upper()
    home = home.upper()

    games = scoreboard.get(
        "games",
        []
    )

    for game in games:
        away_team = (
            game.get(
                "awayTeam",
                {}
            )
            .get(
                "abbrev",
                ""
            )
            .upper()
        )

        home_team = (
            game.get(
                "homeTeam",
                {}
            )
            .get(
                "abbrev",
                ""
            )
            .upper()
        )

        if (
            away_team == away
            and home_team == home
        ):
            return game

    return None


def is_final(game):
    state = clean_str(
        game.get(
            "gameState"
        )
    ).upper()

    return state in {
        "FINAL",
        "OFF",
    }


def american_win_profit(odds):
    odds = float(odds)

    if odds > 0:
        return odds / 100.0

    if odds < 0:
        return 100.0 / abs(odds)

    raise ValueError(
        "American odds cannot be zero."
    )


def grade_h2h(
    selection,
    away,
    home,
    away_score,
    home_score,
):
    selection = selection.upper()

    if away_score == home_score:
        return "PUSH"

    winner = (
        away.upper()
        if away_score > home_score
        else home.upper()
    )

    return (
        "WIN"
        if selection == winner
        else "LOSS"
    )


def grade_spread(
    selection,
    line,
    away,
    home,
    away_score,
    home_score,
):
    selection = selection.upper()
    line = float(line)

    if selection == away.upper():
        adjusted = (
            away_score + line
        )

        opponent = home_score

    elif selection == home.upper():
        adjusted = (
            home_score + line
        )

        opponent = away_score

    else:
        raise ValueError(
            f"Unknown spread selection "
            f"{selection}"
        )

    if adjusted > opponent:
        return "WIN"

    if adjusted < opponent:
        return "LOSS"

    return "PUSH"


def grade_total(
    selection,
    line,
    away_score,
    home_score,
):
    selection = (
        selection
        .strip()
        .upper()
    )

    line = float(line)

    actual = (
        away_score
        + home_score
    )

    if actual == line:
        return "PUSH"

    if selection.startswith("OVER"):
        return (
            "WIN"
            if actual > line
            else "LOSS"
        )

    if selection.startswith("UNDER"):
        return (
            "WIN"
            if actual < line
            else "LOSS"
        )

    raise ValueError(
        f"Unknown total selection "
        f"{selection}"
    )


def grade_market(
    market,
    selection,
    line,
    away,
    home,
    away_score,
    home_score,
):
    market = (
        clean_str(market)
        .lower()
    )

    if market in {
        "h2h",
        "moneyline",
        "ml",
    }:
        return grade_h2h(
            selection,
            away,
            home,
            away_score,
            home_score,
        )

    if market in {
        "spreads",
        "spread",
        "puck_line",
        "puckline",
    }:
        return grade_spread(
            selection,
            line,
            away,
            home,
            away_score,
            home_score,
        )

    if market in {
        "totals",
        "total",
    }:
        return grade_total(
            selection,
            line,
            away_score,
            home_score,
        )

    raise ValueError(
        f"Unsupported market: {market}"
    )


def profit_for_result(
    result,
    odds,
):
    if result == "WIN":
        return american_win_profit(
            odds
        )

    if result == "LOSS":
        return -1.0

    if result == "PUSH":
        return 0.0

    return np.nan


def ensure_columns(df):
    defaults = {
        "nhl_game_id": pd.NA,
        "game_state": pd.NA,
        "final_away_score": np.nan,
        "final_home_score": np.nan,
        "graded_at_utc": pd.NA,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Make result string-safe.
    df["result"] = (
        df["result"]
        .astype("object")
    )

    return df


def build_summary(df):
    graded = df[
        df["result"].isin(
            [
                "WIN",
                "LOSS",
                "PUSH",
            ]
        )
    ].copy()

    if graded.empty:
        return pd.DataFrame(
            [
                {
                    "graded_signals": 0,
                    "wins": 0,
                    "losses": 0,
                    "pushes": 0,
                    "risked_units": 0.0,
                    "profit_units": 0.0,
                    "roi": np.nan,
                }
            ]
        )

    wins = int(
        (graded["result"] == "WIN")
        .sum()
    )

    losses = int(
        (graded["result"] == "LOSS")
        .sum()
    )

    pushes = int(
        (graded["result"] == "PUSH")
        .sum()
    )

    risked = float(
        wins + losses
    )

    profit = float(
        graded["profit_units"]
        .sum()
    )

    roi = (
        profit / risked
        if risked > 0
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "graded_signals":
                    len(graded),

                "wins":
                    wins,

                "losses":
                    losses,

                "pushes":
                    pushes,

                "risked_units":
                    risked,

                "profit_units":
                    profit,

                "roi":
                    roi,
            }
        ]
    )


def main():
    print()
    print(
        "NHL SHADOW GRADER v0.7.8"
    )
    print(
        "------------------------"
    )

    if not LOG.exists():
        raise FileNotFoundError(
            f"Missing shadow log: {LOG}"
        )

    df = pd.read_csv(LOG)

    df = ensure_columns(df)

    # Back up before modifying.
    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    backup = LOG.with_name(
        f"{LOG.stem}_backup_"
        f"{timestamp}.csv"
    )

    shutil.copy2(
        LOG,
        backup,
    )

    print(
        f"Signals: {len(df)}"
    )

    print(
        f"Backup:  {backup}"
    )

    scoreboards = {}

    newly_graded = 0
    pending = 0
    errors = 0

    for idx, row in df.iterrows():
        # Don't re-grade completed rows.
        existing_result = (
            clean_str(
                row.get(
                    "result"
                )
            )
            .upper()
        )

        if existing_result in {
            "WIN",
            "LOSS",
            "PUSH",
        }:
            continue

        date = clean_str(
            row["date"]
        )

        away = clean_str(
            row["away"]
        ).upper()

        home = clean_str(
            row["home"]
        ).upper()

        print()
        print(
            f"{date} "
            f"{away} @ {home}"
        )

        try:
            if date not in scoreboards:
                scoreboards[
                    date
                ] = fetch_scoreboard(
                    date
                )

            game = find_game(
                scoreboards[date],
                away,
                home,
            )

            if game is None:
                print(
                    "  NHL game not found "
                    "for this date."
                )

                pending += 1
                continue

            game_id = game.get(
                "id"
            )

            state = clean_str(
                game.get(
                    "gameState"
                )
            ).upper()

            df.loc[
                idx,
                "nhl_game_id",
            ] = game_id

            df.loc[
                idx,
                "game_state",
            ] = state

            print(
                f"  Game state: {state}"
            )

            if not is_final(game):
                print(
                    "  Not final -> pending"
                )

                pending += 1
                continue

            away_score = int(
                game[
                    "awayTeam"
                ][
                    "score"
                ]
            )

            home_score = int(
                game[
                    "homeTeam"
                ][
                    "score"
                ]
            )

            result = grade_market(
                row["market"],
                clean_str(
                    row["selection"]
                ),
                row["line"],
                away,
                home,
                away_score,
                home_score,
            )

            profit = (
                profit_for_result(
                    result,
                    row["odds"],
                )
            )

            graded_at = (
                datetime.now(
                    timezone.utc
                )
                .isoformat()
            )

            df.loc[
                idx,
                "final_away_score",
            ] = away_score

            df.loc[
                idx,
                "final_home_score",
            ] = home_score

            df.loc[
                idx,
                "result",
            ] = result

            df.loc[
                idx,
                "profit_units",
            ] = profit

            df.loc[
                idx,
                "graded_at_utc",
            ] = graded_at

            newly_graded += 1

            print(
                f"  Final: "
                f"{away} {away_score} - "
                f"{home} {home_score}"
            )

            print(
                f"  {row['selection']} "
                f"{int(row['odds']):+d}: "
                f"{result}"
            )

            print(
                f"  Profit: "
                f"{profit:+.3f}u"
            )

        except Exception as exc:
            errors += 1

            print(
                f"  ERROR: {exc}"
            )

    # Save updated log.
    df.to_csv(
        LOG,
        index=False,
    )

    summary = build_summary(
        df
    )

    summary.to_csv(
        SUMMARY,
        index=False,
    )

    print()
    print("=" * 60)
    print("SHADOW PERFORMANCE")
    print("=" * 60)

    s = summary.iloc[0]

    print(
        f"Newly graded: "
        f"{newly_graded}"
    )

    print(
        f"Pending:      "
        f"{pending}"
    )

    print(
        f"Errors:       "
        f"{errors}"
    )

    print()

    print(
        f"Record: "
        f"{int(s['wins'])}-"
        f"{int(s['losses'])}-"
        f"{int(s['pushes'])}"
    )

    print(
        f"Risked: "
        f"{s['risked_units']:.2f}u"
    )

    print(
        f"Profit: "
        f"{s['profit_units']:+.3f}u"
    )

    if pd.notna(
        s["roi"]
    ):
        print(
            f"ROI: "
            f"{s['roi']:+.2%}"
        )

    else:
        print(
            "ROI: pending"
        )

    print()
    print("UPDATED LOG")
    print("-----------")

    show_cols = [
        "date",
        "away",
        "home",
        "market",
        "selection",
        "line",
        "odds",
        "book",
        "result",
        "profit_units",
        "final_away_score",
        "final_home_score",
        "game_state",
        "placed",
    ]

    print(
        df[
            show_cols
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("SAVED")
    print("-----")
    print(LOG)
    print(SUMMARY)


if __name__ == "__main__":
    main()
