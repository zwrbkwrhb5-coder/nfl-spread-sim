from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def labor_day(year: int) -> date:
    d = date(year, 9, 1)
    return d + timedelta(days=(7 - d.weekday()) % 7)


def regular_season_week(season: int, on_date: date) -> int:
    week1_board_start = labor_day(season) + timedelta(days=1)
    if on_date < week1_board_start:
        return 1
    week = ((on_date - week1_board_start).days // 7) + 1
    return max(1, min(18, week))


def season_for_date(on_date: date) -> int:
    if on_date.month <= 2:
        return on_date.year - 1
    return on_date.year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD in Eastern Time")
    ap.add_argument("--season", type=int)
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()

    if args.date:
        on_date = date.fromisoformat(args.date)
    else:
        on_date = datetime.now(ET).date()

    season = args.season or season_for_date(on_date)
    current = regular_season_week(season, on_date)
    next_week = current + 1 if current < 18 else None

    payload = {
        "season": season,
        "dateEastern": on_date.isoformat(),
        "currentWeek": current,
        "nextWeek": next_week,
    }

    if args.shell:
        print(f"SEASON={season}")
        print(f"CURRENT_WEEK={current}")
        print(f"NEXT_WEEK={next_week or 0}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
