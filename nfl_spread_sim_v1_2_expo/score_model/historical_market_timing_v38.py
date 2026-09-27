from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare_columns(df: pd.DataFrame, live_col: str, closing_col: str):
    a = pd.to_numeric(df[live_col], errors="coerce")
    b = pd.to_numeric(df[closing_col], errors="coerce")

    mask = a.notna() & b.notna()
    diff = (a[mask] - b[mask]).abs()

    if mask.sum() == 0:
        return {
            "field": live_col,
            "closing_field": closing_col,
            "paired_rows": 0,
            "exact_match_rows": 0,
            "exact_match_rate": np.nan,
            "different_rows": 0,
            "mean_abs_difference": np.nan,
            "max_abs_difference": np.nan,
        }

    return {
        "field": live_col,
        "closing_field": closing_col,
        "paired_rows": int(mask.sum()),
        "exact_match_rows": int((diff < 1e-12).sum()),
        "exact_match_rate": float((diff < 1e-12).mean()),
        "different_rows": int((diff >= 1e-12).sum()),
        "mean_abs_difference": float(diff.mean()),
        "max_abs_difference": float(diff.max()),
    }


def audit_market(df: pd.DataFrame):
    required = [
        "home_spread",
        "closing_home_spread",
        "total_line",
        "closing_total",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing timing-audit columns: {missing}")

    rows = [
        compare_columns(df, "home_spread", "closing_home_spread"),
        compare_columns(df, "total_line", "closing_total"),
    ]
    out = pd.DataFrame(rows)

    rates = out["exact_match_rate"].dropna().to_numpy(float)

    if len(rates) == 2 and np.all(rates >= 0.98):
        classification = "closing_alias"
    elif len(rates) and np.all(rates >= 0.90):
        classification = "mostly_closing"
    else:
        classification = "mixed_or_unknown"

    return out, classification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--market-csv",
        default="data/historical_market.csv",
    )
    ap.add_argument(
        "--output-dir",
        default="artifacts/market_timing_v38",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.market_csv)
    table, classification = audit_market(df)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    table.to_csv(out / "timing_audit.csv", index=False)
    (out / "timing_classification.json").write_text(
        json.dumps(
            {
                "classification": classification,
                "rows": int(len(df)),
                "note": (
                    "closing_alias means the generic market fields are "
                    "numerically identical to the explicit closing fields "
                    "on at least 98% of paired rows. It does not identify "
                    "book, timestamp, or exact collection methodology."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nV3.8 HISTORICAL MARKET TIMING AUDIT")
    print(table.to_string(index=False))
    print("\nClassification:", classification)

    if classification == "closing_alias":
        print(
            "\nTreat home_spread and total_line as closing-line aliases "
            "for model-validation purposes."
        )
    elif classification == "mostly_closing":
        print(
            "\nThe generic fields mostly match closing fields. "
            "Inspect differing rows before using them as live-time inputs."
        )
    else:
        print(
            "\nHistorical timing is mixed/unknown. "
            "Do not treat these fields as a consistent closing market."
        )


if __name__ == "__main__":
    main()
