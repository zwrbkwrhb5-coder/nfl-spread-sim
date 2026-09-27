from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


QB_METRICS = [
    "qb_epa_per_dropback",
    "qb_success_rate",
    "qb_cpoe",
    "qb_sack_rate",
    "qb_interception_rate",
    "qb_explosive_pass_rate",
]

NEUTRAL_TARGETS = {
    "qb_epa_per_dropback": 0.0,
    "qb_success_rate": 0.45,
    "qb_cpoe": 0.0,
    "qb_sack_rate": 0.07,
    "qb_interception_rate": 0.025,
    "qb_explosive_pass_rate": 0.12,
}


def add_week_key(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["week_key"] = (
        pd.to_numeric(d["season"], errors="raise").astype(int) * 100
        + pd.to_numeric(d["week"], errors="raise").astype(int)
    )
    return d


def pooled_prior_target(prior: pd.DataFrame, metric: str) -> tuple[float, int]:
    vals = []
    for side in ["home", "away"]:
        c = f"{side}_pre_{metric}"
        if c in prior.columns:
            vals.append(pd.to_numeric(prior[c], errors="coerce"))

    if not vals:
        return float(NEUTRAL_TARGETS[metric]), 0

    x = pd.concat(vals, ignore_index=True).dropna()
    if x.empty:
        return float(NEUTRAL_TARGETS[metric]), 0

    return float(x.mean()), int(len(x))


def apply_past_only_qb_shrinkage(
    df: pd.DataFrame,
    scale_dropbacks: float = 100.0,
) -> pd.DataFrame:
    """
    Create leakage-safe QB features.

    For every season/week:
      - league shrink targets use ONLY rows from earlier weeks
      - no same-week row can affect another row in that week
      - home/away use one pooled league target
      - rookie / low-sample QBs shrink toward that prior-only target

    Original raw QB features remain untouched. New columns end in `_v28`.
    """
    d = add_week_key(df)
    d = d.sort_values(["week_key", "game_id"]).copy()

    for metric in QB_METRICS:
        d[f"qb_target_{metric}_v28"] = np.nan
        d[f"qb_target_n_{metric}_v28"] = 0

    d["qb_target_source_max_week_key_v28"] = np.nan

    for wk in sorted(int(x) for x in d["week_key"].unique()):
        mask = d["week_key"].eq(wk)
        prior = d[d["week_key"] < wk]

        d.loc[mask, "qb_target_source_max_week_key_v28"] = (
            float(prior["week_key"].max()) if not prior.empty else np.nan
        )

        for metric in QB_METRICS:
            target, n = pooled_prior_target(prior, metric)
            d.loc[mask, f"qb_target_{metric}_v28"] = target
            d.loc[mask, f"qb_target_n_{metric}_v28"] = n

    for side in ["home", "away"]:
        db_col = f"{side}_qb_prior_dropbacks"
        if db_col not in d.columns:
            raise ValueError(f"Missing required QB sample column: {db_col}")

        n = pd.to_numeric(d[db_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        w = n / (n + float(scale_dropbacks))

        qb_id_col = f"{side}_qb_id"
        has_qb = (
            d[qb_id_col].notna()
            if qb_id_col in d.columns
            else pd.Series(True, index=d.index)
        )

        for metric in QB_METRICS:
            raw_col = f"{side}_pre_{metric}"
            if raw_col not in d.columns:
                raise ValueError(f"Missing required QB feature: {raw_col}")

            target_col = f"qb_target_{metric}_v28"
            out_col = f"{raw_col}_v28"

            raw = pd.to_numeric(d[raw_col], errors="coerce")
            target = pd.to_numeric(d[target_col], errors="coerce")

            shrunk = target + w * (raw - target)

            # A matched QB with no usable form is treated as a true low-sample
            # QB and receives the prior-only league target.
            shrunk = shrunk.where(raw.notna(), target)
            shrunk = shrunk.where(has_qb, np.nan)

            d[out_col] = shrunk

    for metric in QB_METRICS:
        d[f"diff_{metric}_v28"] = (
            d[f"home_pre_{metric}_v28"]
            - d[f"away_pre_{metric}_v28"]
        )
        d[f"sum_{metric}_v28"] = (
            d[f"home_pre_{metric}_v28"]
            + d[f"away_pre_{metric}_v28"]
        )

    return d


def audit_no_future_targets(df: pd.DataFrame) -> dict:
    d = add_week_key(df) if "week_key" not in df.columns else df.copy()

    bad = d[
        d["qb_target_source_max_week_key_v28"].notna()
        & (
            d["qb_target_source_max_week_key_v28"]
            >= d["week_key"]
        )
    ]

    return {
        "rows": int(len(d)),
        "future_or_same_week_target_rows": int(len(bad)),
        "past_only": bool(len(bad) == 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--training-csv",
        default="artifacts/dual_market/training_games_qb_context_v24.csv",
    )
    ap.add_argument(
        "--output",
        default="artifacts/dual_market/training_games_qb_safe_context_v28.csv",
    )
    ap.add_argument("--shrinkage-dropbacks", type=float, default=100.0)
    args = ap.parse_args()

    df = pd.read_csv(args.training_csv)
    out = apply_past_only_qb_shrinkage(
        df,
        scale_dropbacks=args.shrinkage_dropbacks,
    )

    audit = audit_no_future_targets(out)
    if not audit["past_only"]:
        raise RuntimeError(f"QB target leakage audit failed: {audit}")

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)

    print(f"Saved leakage-safe QB table: {p}")
    print(f"Rows: {len(out):,}")
    print(f"Shrinkage scale: {args.shrinkage_dropbacks:g} prior dropbacks")
    print(f"Future/same-week target rows: {audit['future_or_same_week_target_rows']}")
    print("Past-only QB shrinkage: true")


if __name__ == "__main__":
    main()
