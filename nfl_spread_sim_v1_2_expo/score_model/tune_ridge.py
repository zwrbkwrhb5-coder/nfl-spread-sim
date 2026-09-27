from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_ALPHAS = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def evaluate_alpha(df, feature_cols, target, alpha, first_test_season):
    rows = []
    for season in sorted(s for s in df["season"].unique() if s >= first_test_season):
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        if train.empty or test.empty:
            continue
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        pipe.fit(train[feature_cols], train[target])
        pred = pipe.predict(test[feature_cols])
        rows.append({
            "season": int(season),
            "n": int(len(test)),
            "mae": float(mean_absolute_error(test[target], pred)),
            "rmse": float(np.sqrt(mean_squared_error(test[target], pred))),
        })
    if not rows:
        return None
    rdf = pd.DataFrame(rows)
    w = rdf["n"] / rdf["n"].sum()
    return {
        "alpha": float(alpha),
        "weighted_mae": float((rdf["mae"] * w).sum()),
        "weighted_rmse": float((rdf["rmse"] * w).sum()),
        "by_season": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", required=True)
    ap.add_argument("--target", choices=["margin", "total"], required=True)
    ap.add_argument("--first-test-season", type=int, default=2022)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.training_csv)
    target_col = "actual_margin" if args.target == "margin" else "actual_total"
    excluded = {
        "game_id", "season", "week", "home_team", "away_team",
        "actual_margin", "actual_total", "home_score", "away_score",
    }
    feature_cols = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    df = df.dropna(subset=feature_cols + [target_col])

    results = []
    for alpha in DEFAULT_ALPHAS:
        r = evaluate_alpha(
            df, feature_cols, target_col, alpha, args.first_test_season
        )
        if r:
            results.append(r)

    results = sorted(results, key=lambda x: (x["weighted_mae"], x["weighted_rmse"]))
    out = {
        "target": args.target,
        "best_alpha": results[0]["alpha"] if results else None,
        "results": results,
    }

    output = args.output or f"artifacts/ridge_tuning_{args.target}.json"
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
