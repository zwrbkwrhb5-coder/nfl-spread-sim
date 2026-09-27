from __future__ import annotations

import argparse
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=3)
    ap.add_argument("--max-age-minutes", type=float, default=30.0)
    ap.add_argument("--snapshot-label", default="current")
    args = ap.parse_args()

    all_candidates = (
        f"artifacts/live/"
        f"week{args.week}_{args.season}_all_candidates_v31.csv"
    )
    best = (
        f"artifacts/live/"
        f"week{args.week}_{args.season}_best_v31.csv"
    )
    shadow = (
        f"artifacts/live/"
        f"week{args.week}_{args.season}_totals_shadow_v38.csv"
    )

    subprocess.run(
        [
            "python",
            "-m",
            "score_model.multi_book_live_v31",
            "--season",
            str(args.season),
            "--week",
            str(args.week),
            "--max-age-minutes",
            str(args.max_age_minutes),
            "--all-candidates-output",
            all_candidates,
            "--best-output",
            best,
        ],
        check=True,
    )

    subprocess.run(
        [
            "python",
            "-m",
            "score_model.live_totals_shadow_v38",
            "--live-all-candidates",
            all_candidates,
            "--snapshot-label",
            args.snapshot_label,
            "--output",
            shadow,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
