from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from score_model.export_mobile_v50 import build_games, optional_csv, required_csv


def load_app_snapshot(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing current mobile snapshot: {p}")

    text = p.read_text(encoding="utf-8")
    marker = "export const APP_DATA = "
    if marker not in text:
        raise ValueError(f"Could not find APP_DATA in {p}")

    payload = text.split(marker, 1)[1]
    suffix = " satisfies AppSnapshot;"
    if suffix in payload:
        payload = payload.rsplit(suffix, 1)[0]
    payload = payload.strip().rstrip(";")
    return json.loads(payload)


def build_game_results_from_snapshot(snapshot):
    existing = snapshot.get("gameResults") or []
    if existing:
        return existing

    by_game = {}
    for r in snapshot.get("forwardPicks", []):
        gid = str(r.get("gameId", ""))
        if not gid:
            continue
        if r.get("actualMargin") is None and r.get("actualTotal") is None:
            continue
        by_game[gid] = {
            "gameId": gid,
            "actualMargin": r.get("actualMargin"),
            "actualTotal": r.get("actualTotal"),
            "awayScore": r.get("awayScore"),
            "homeScore": r.get("homeScore"),
        }
    return list(by_game.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--current-week", type=int, default=3)
    ap.add_argument("--next-week", type=int, default=4)
    ap.add_argument("--current-snapshot", default="mobile/generatedData.ts")
    ap.add_argument("--next-best", default="artifacts/live/week4_2026_best_v31.csv")
    ap.add_argument("--next-all-candidates", default="artifacts/live/week4_2026_all_candidates_v31.csv")
    ap.add_argument("--output", default="mobile/generatedData.ts")
    args = ap.parse_args()

    snapshot = load_app_snapshot(args.current_snapshot)
    current_games = [
        g for g in snapshot.get("games", [])
        if int(g.get("week", args.current_week)) == args.current_week
    ]
    if not current_games:
        raise RuntimeError(
            f"The existing mobile snapshot does not contain Week {args.current_week} games. "
            "Do not rerun the old refresh script; send the terminal output instead."
        )

    next_best = required_csv(args.next_best, "next-week best file")
    next_candidates = optional_csv(args.next_all_candidates)
    next_games = build_games(next_best, next_candidates, args.season, args.next_week)

    # Preserve the current-week app snapshot exactly as it was, and append the
    # newly generated next-week model output.
    games = current_games + next_games
    games.sort(key=lambda x: (int(x.get("week", 999)), x.get("kickoffAt") or "9999", x.get("id") or ""))

    snapshot["generatedAt"] = pd.Timestamp.now(tz="UTC").isoformat()
    snapshot["season"] = args.season
    snapshot["week"] = args.current_week
    snapshot["currentWeek"] = args.current_week
    snapshot["nextWeek"] = args.next_week
    snapshot["availableWeeks"] = [args.current_week, args.next_week]
    snapshot["games"] = games
    snapshot["gameResults"] = build_game_results_from_snapshot(snapshot)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        'import type { AppSnapshot } from "./types";\n\n'
        + "export const APP_DATA = "
        + json.dumps(snapshot, indent=2)
        + " satisfies AppSnapshot;\n",
        encoding="utf-8",
    )

    q = sum(1 for g in next_games if g.get("qualifies"))
    print("V5.0.1 WEEK 4 RECOVERY EXPORT")
    print(f"Preserved Week {args.current_week}: {len(current_games)} games")
    print(f"Added Week {args.next_week}: {len(next_games)} games")
    print(f"Week {args.next_week} games with qualifying market(s): {q}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
