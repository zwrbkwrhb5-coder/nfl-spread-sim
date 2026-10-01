from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlencode
import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd


LOG = Path(
    "artifacts/nhl/live/nhl_shadow_signals.csv"
)

SNAPSHOTS = Path(
    "artifacts/nhl/live/nhl_clv_snapshots.csv"
)

ODDS_API = (
    "https://api.the-odds-api.com/v4/"
    "sports/icehockey_nhl/odds/"
)

NHL_SCORE_API = (
    "https://api-web.nhle.com/v1/score"
)

# Only spend Odds API credits when a tracked
# game is getting reasonably close to puck drop.
CAPTURE_WINDOW_MIN = 90


TEAM_NAMES = {
    "ANA": {
        "Anaheim Ducks",
    },
    "BOS": {
        "Boston Bruins",
    },
    "BUF": {
        "Buffalo Sabres",
    },
    "CAR": {
        "Carolina Hurricanes",
    },
    "CBJ": {
        "Columbus Blue Jackets",
    },
    "CGY": {
        "Calgary Flames",
    },
    "CHI": {
        "Chicago Blackhawks",
    },
    "COL": {
        "Colorado Avalanche",
    },
    "DAL": {
        "Dallas Stars",
    },
    "DET": {
        "Detroit Red Wings",
    },
    "EDM": {
        "Edmonton Oilers",
    },
    "FLA": {
        "Florida Panthers",
    },
    "LAK": {
        "Los Angeles Kings",
    },
    "MIN": {
        "Minnesota Wild",
    },
    "MTL": {
        "Montreal Canadiens",
    },
    "NJD": {
        "New Jersey Devils",
    },
    "NSH": {
        "Nashville Predators",
    },
    "NYI": {
        "New York Islanders",
    },
    "NYR": {
        "New York Rangers",
    },
    "OTT": {
        "Ottawa Senators",
    },
    "PHI": {
        "Philadelphia Flyers",
    },
    "PIT": {
        "Pittsburgh Penguins",
    },
    "SEA": {
        "Seattle Kraken",
    },
    "SJS": {
        "San Jose Sharks",
    },
    "STL": {
        "St Louis Blues",
        "St. Louis Blues",
    },
    "TBL": {
        "Tampa Bay Lightning",
    },
    "TOR": {
        "Toronto Maple Leafs",
    },
    "UTA": {
        "Utah Mammoth",
        "Utah Hockey Club",
    },
    "VAN": {
        "Vancouver Canucks",
    },
    "VGK": {
        "Vegas Golden Knights",
    },
    "WPG": {
        "Winnipeg Jets",
    },
    "WSH": {
        "Washington Capitals",
    },
}


def norm(x):
    if pd.isna(x):
        return ""

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(x).lower(),
    )


NAME_TO_ABBR = {}

for abbr, names in TEAM_NAMES.items():
    for name in names:
        NAME_TO_ABBR[
            norm(name)
        ] = abbr


def clean(x):
    if pd.isna(x):
        return ""

    return str(x).strip()


def fetch_json(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0 NHL-CLV-capture"
        },
    )

    with urllib.request.urlopen(
        req,
        timeout=30,
    ) as response:
        return json.load(response)


def get_nhl_scoreboard(date):
    return fetch_json(
        f"{NHL_SCORE_API}/{date}"
    )


def find_nhl_game(
    scoreboard,
    away,
    home,
):
    away = away.upper()
    home = home.upper()

    for game in scoreboard.get(
        "games",
        []
    ):
        a = (
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

        h = (
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

        if a == away and h == home:
            return game

    return None


def parse_utc(value):
    return datetime.fromisoformat(
        value.replace(
            "Z",
            "+00:00",
        )
    )


def american_to_decimal(odds):
    odds = float(odds)

    if odds > 0:
        return 1.0 + odds / 100.0

    if odds < 0:
        return 1.0 + 100.0 / abs(odds)

    return np.nan


def american_implied(odds):
    odds = float(odds)

    if odds > 0:
        return 100.0 / (
            odds + 100.0
        )

    if odds < 0:
        return abs(odds) / (
            abs(odds) + 100.0
        )

    return np.nan


def market_key(value):
    value = clean(value).lower()

    if value in {
        "h2h",
        "moneyline",
        "ml",
    }:
        return "h2h"

    if value in {
        "spread",
        "spreads",
        "puckline",
        "puck_line",
    }:
        return "spreads"

    if value in {
        "total",
        "totals",
    }:
        return "totals"

    return value


def selection_abbr(name):
    return NAME_TO_ABBR.get(
        norm(name)
    )


def find_odds_event(
    events,
    away,
    home,
):
    for event in events:
        a = selection_abbr(
            event.get(
                "away_team",
                ""
            )
        )

        h = selection_abbr(
            event.get(
                "home_team",
                ""
            )
        )

        if (
            a == away.upper()
            and h == home.upper()
        ):
            return event

    return None


def find_bookmaker(
    event,
    wanted_book,
):
    wanted = norm(
        wanted_book
    )

    for book in event.get(
        "bookmakers",
        []
    ):
        if norm(
            book.get(
                "title",
                ""
            )
        ) == wanted:
            return book

        if norm(
            book.get(
                "key",
                ""
            )
        ) == wanted:
            return book

    return None


def find_market(
    bookmaker,
    wanted_market,
):
    for market in bookmaker.get(
        "markets",
        []
    ):
        if (
            market.get("key")
            == wanted_market
        ):
            return market

    return None


def find_outcome(
    market,
    market_type,
    selection,
):
    selection = clean(
        selection
    ).upper()

    for outcome in market.get(
        "outcomes",
        []
    ):
        name = clean(
            outcome.get(
                "name"
            )
        )

        if market_type in {
            "h2h",
            "spreads",
        }:
            if (
                selection_abbr(name)
                == selection
            ):
                return outcome

        elif market_type == "totals":
            if (
                selection.startswith(
                    "OVER"
                )
                and name.upper()
                == "OVER"
            ):
                return outcome

            if (
                selection.startswith(
                    "UNDER"
                )
                and name.upper()
                == "UNDER"
            ):
                return outcome

    return None


def signal_key(row):
    line = (
        ""
        if pd.isna(row.get("line"))
        else str(row.get("line"))
    )

    return "|".join(
        [
            clean(row.get("date")),
            clean(row.get("away")),
            clean(row.get("home")),
            clean(row.get("market")),
            clean(row.get("selection")),
            line,
            clean(row.get("odds")),
            clean(row.get("book")),
        ]
    )


def ensure_log_columns(df):
    defaults = {
        "scheduled_start_utc":
            pd.NA,

        "close_captured_at_utc":
            pd.NA,

        "close_book":
            pd.NA,

        "close_line":
            np.nan,

        "close_odds":
            np.nan,

        "close_break_even":
            np.nan,

        "clv_probability_points":
            np.nan,

        "clv_decimal_pct":
            np.nan,

        "clv_line":
            np.nan,

        "clv_status":
            pd.NA,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    return df


def load_snapshots():
    if not SNAPSHOTS.exists():
        return pd.DataFrame()

    return pd.read_csv(
        SNAPSHOTS
    )


def save_snapshot(row):
    old = load_snapshots()

    new = pd.DataFrame(
        [row]
    )

    combined = pd.concat(
        [
            old,
            new,
        ],
        ignore_index=True,
    )

    combined.to_csv(
        SNAPSHOTS,
        index=False,
    )


def latest_snapshot(
    snapshots,
    key,
    start,
):
    if snapshots.empty:
        return None

    part = snapshots[
        snapshots[
            "signal_key"
        ] == key
    ].copy()

    if part.empty:
        return None

    part[
        "captured_dt"
    ] = pd.to_datetime(
        part[
            "captured_at_utc"
        ],
        utc=True,
        errors="coerce",
    )

    start_ts = pd.Timestamp(
        start
    )

    part = part[
        part["captured_dt"]
        <= start_ts
    ]

    if part.empty:
        return None

    return (
        part.sort_values(
            "captured_dt"
        )
        .iloc[-1]
    )


def line_clv(
    market,
    selection,
    entry_line,
    close_line,
):
    if (
        pd.isna(entry_line)
        or pd.isna(close_line)
    ):
        return np.nan

    entry_line = float(
        entry_line
    )

    close_line = float(
        close_line
    )

    if market == "spreads":
        # Example:
        # bet +1.5, closes +1.0 => +0.5 CLV
        # bet -1.5, closes -2.0 => +0.5 CLV
        return (
            entry_line
            - close_line
        )

    if market == "totals":
        selection = (
            clean(selection)
            .upper()
        )

        if selection.startswith(
            "UNDER"
        ):
            return (
                entry_line
                - close_line
            )

        if selection.startswith(
            "OVER"
        ):
            return (
                close_line
                - entry_line
            )

    return np.nan


def update_from_snapshot(
    df,
    idx,
    snap,
    final_status,
):
    entry_odds = float(
        df.loc[
            idx,
            "odds",
        ]
    )

    close_odds = float(
        snap[
            "snapshot_odds"
        ]
    )

    entry_prob = (
        american_implied(
            entry_odds
        )
    )

    close_prob = (
        american_implied(
            close_odds
        )
    )

    entry_decimal = (
        american_to_decimal(
            entry_odds
        )
    )

    close_decimal = (
        american_to_decimal(
            close_odds
        )
    )

    df.loc[
        idx,
        "close_captured_at_utc",
    ] = snap[
        "captured_at_utc"
    ]

    df.loc[
        idx,
        "close_book",
    ] = snap[
        "book"
    ]

    df.loc[
        idx,
        "close_line",
    ] = snap[
        "snapshot_line"
    ]

    df.loc[
        idx,
        "close_odds",
    ] = close_odds

    df.loc[
        idx,
        "close_break_even",
    ] = close_prob

    # Positive means market moved toward
    # our side after we logged the signal.
    df.loc[
        idx,
        "clv_probability_points",
    ] = (
        close_prob
        - entry_prob
    ) * 100.0

    market = market_key(
        df.loc[
            idx,
            "market",
        ]
    )

    entry_line = df.loc[
        idx,
        "line",
    ]

    close_line = snap[
        "snapshot_line"
    ]

    df.loc[
        idx,
        "clv_line",
    ] = line_clv(
        market,
        df.loc[
            idx,
            "selection",
        ],
        entry_line,
        close_line,
    )

    # Price-only CLV is clean for h2h.
    # For spreads/totals only calculate
    # when the line itself did not move.
    same_line = (
        market == "h2h"
        or (
            pd.notna(entry_line)
            and pd.notna(close_line)
            and np.isclose(
                float(entry_line),
                float(close_line),
            )
        )
    )

    if same_line:
        df.loc[
            idx,
            "clv_decimal_pct",
        ] = (
            (
                entry_decimal
                / close_decimal
            )
            - 1.0
        ) * 100.0

    else:
        df.loc[
            idx,
            "clv_decimal_pct",
        ] = np.nan

    df.loc[
        idx,
        "clv_status",
    ] = final_status


def main():
    print()
    print(
        "NHL CLV CAPTURE v0.7.9"
    )
    print(
        "----------------------"
    )

    api_key = os.environ.get(
        "THE_ODDS_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "THE_ODDS_API_KEY is not set."
        )

    if not LOG.exists():
        raise FileNotFoundError(
            LOG
        )

    df = pd.read_csv(
        LOG
    )

    df = ensure_log_columns(
        df
    )

    now = datetime.now(
        timezone.utc
    )

    scoreboards = {}
    game_info = {}

    eligible = []

    print(
        f"Signals: {len(df)}"
    )

    # ----------------------------------------------
    # Resolve NHL start time/state
    # ----------------------------------------------

    for idx, row in df.iterrows():
        date = clean(
            row["date"]
        )

        away = clean(
            row["away"]
        ).upper()

        home = clean(
            row["home"]
        ).upper()

        if date not in scoreboards:
            scoreboards[
                date
            ] = get_nhl_scoreboard(
                date
            )

        game = find_nhl_game(
            scoreboards[date],
            away,
            home,
        )

        if game is None:
            print()
            print(
                f"{away} @ {home}: "
                "NHL game not found"
            )
            continue

        start_text = game.get(
            "startTimeUTC"
        )

        if not start_text:
            print()
            print(
                f"{away} @ {home}: "
                "missing startTimeUTC"
            )
            continue

        start = parse_utc(
            start_text
        )

        state = clean(
            game.get(
                "gameState"
            )
        ).upper()

        minutes_to_start = (
            start - now
        ).total_seconds() / 60.0

        df.loc[
            idx,
            "scheduled_start_utc",
        ] = start.isoformat()

        key = signal_key(
            row
        )

        game_info[idx] = {
            "key":
                key,

            "start":
                start,

            "state":
                state,

            "minutes_to_start":
                minutes_to_start,

            "away":
                away,

            "home":
                home,
        }

        print()
        print(
            f"{away} @ {home}"
        )

        print(
            f"  State: {state}"
        )

        print(
            "  Minutes to start: "
            f"{minutes_to_start:+.1f}"
        )

        if (
            state in {
                "FUT",
                "PRE",
            }
            and 0
            <= minutes_to_start
            <= CAPTURE_WINDOW_MIN
        ):
            eligible.append(
                idx
            )

            print(
                "  Eligible for "
                "pregame odds snapshot"
            )

    # ----------------------------------------------
    # One Odds API call for all eligible games
    # ----------------------------------------------

    if eligible:
        markets = sorted(
            {
                market_key(
                    df.loc[
                        idx,
                        "market",
                    ]
                )
                for idx in eligible
            }
        )

        params = {
            "apiKey":
                api_key,

            "regions":
                "us,us2",

            "markets":
                ",".join(
                    markets
                ),

            "oddsFormat":
                "american",

            "dateFormat":
                "iso",
        }

        url = (
            ODDS_API
            + "?"
            + urlencode(
                params
            )
        )

        print()
        print(
            "Fetching current NHL odds..."
        )

        events = fetch_json(
            url
        )

        captured_at = (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        )

        for idx in eligible:
            row = df.loc[
                idx
            ]

            info = game_info[
                idx
            ]

            event = find_odds_event(
                events,
                info["away"],
                info["home"],
            )

            print()
            print(
                f"Snapshot "
                f"{info['away']} @ "
                f"{info['home']}"
            )

            if event is None:
                print(
                    "  Odds event not found."
                )
                continue

            book = find_bookmaker(
                event,
                row["book"],
            )

            if book is None:
                print(
                    "  Same sportsbook "
                    f"not available: "
                    f"{row['book']}"
                )
                continue

            mk = market_key(
                row["market"]
            )

            market = find_market(
                book,
                mk,
            )

            if market is None:
                print(
                    f"  Market unavailable: "
                    f"{mk}"
                )
                continue

            outcome = find_outcome(
                market,
                mk,
                row["selection"],
            )

            if outcome is None:
                print(
                    "  Selection not found."
                )
                continue

            snapshot_odds = (
                outcome.get(
                    "price"
                )
            )

            snapshot_line = (
                outcome.get(
                    "point",
                    np.nan,
                )
            )

            snapshot = {
                "signal_key":
                    info["key"],

                "date":
                    row["date"],

                "away":
                    info["away"],

                "home":
                    info["home"],

                "market":
                    mk,

                "selection":
                    row["selection"],

                "entry_line":
                    row["line"],

                "entry_odds":
                    row["odds"],

                "book":
                    book.get(
                        "title"
                    ),

                "book_key":
                    book.get(
                        "key"
                    ),

                "snapshot_line":
                    snapshot_line,

                "snapshot_odds":
                    snapshot_odds,

                "captured_at_utc":
                    captured_at,

                "scheduled_start_utc":
                    info[
                        "start"
                    ].isoformat(),

                "minutes_to_start":
                    info[
                        "minutes_to_start"
                    ],
            }

            save_snapshot(
                snapshot
            )

            print(
                f"  {row['selection']} "
                f"{snapshot_line} "
                f"{float(snapshot_odds):+g} "
                f"@ {book.get('title')}"
            )

    else:
        print()
        print(
            "No games currently inside "
            "the CLV capture window."
        )

    # ----------------------------------------------
    # Use latest pregame snapshot for each signal
    # ----------------------------------------------

    snapshots = load_snapshots()

    for idx, info in game_info.items():
        snap = latest_snapshot(
            snapshots,
            info["key"],
            info["start"],
        )

        if snap is None:
            continue

        is_pregame = (
            info["state"]
            in {
                "FUT",
                "PRE",
            }
            and now
            < info["start"]
        )

        status = (
            "PROVISIONAL"
            if is_pregame
            else "FINAL"
        )

        update_from_snapshot(
            df,
            idx,
            snap,
            status,
        )

    df.to_csv(
        LOG,
        index=False,
    )

    print()
    print("=" * 72)
    print("CLV STATUS")
    print("=" * 72)

    cols = [
        "date",
        "away",
        "home",
        "market",
        "selection",
        "line",
        "odds",
        "book",
        "close_line",
        "close_odds",
        "close_book",
        "clv_probability_points",
        "clv_decimal_pct",
        "clv_line",
        "clv_status",
    ]

    print(
        df[cols]
        .to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(LOG)
    print(SNAPSHOTS)


if __name__ == "__main__":
    main()
