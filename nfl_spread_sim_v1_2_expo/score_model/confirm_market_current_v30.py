from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-csv", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument(
        "--i-confirm-lines-are-current",
        action="store_true",
        help=(
            "Required. Use only immediately after manually replacing the CSV "
            "with current sportsbook lines."
        ),
    )
    args = ap.parse_args()

    if not args.i_confirm_lines_are_current:
        raise SystemExit(
            "Refusing to timestamp the file. Re-run with "
            "--i-confirm-lines-are-current only after you manually refresh "
            "the sportsbook lines."
        )

    src = Path(args.market_csv)
    out = Path(args.output) if args.output else src

    d = pd.read_csv(src)
    d["market_updated_at"] = pd.Timestamp.now(tz="UTC").isoformat()

    if "market_source" not in d.columns:
        d["market_source"] = "manual_confirmed"

    d.to_csv(out, index=False)
    print(f"Stamped current market lines: {out}")


if __name__ == "__main__":
    main()
