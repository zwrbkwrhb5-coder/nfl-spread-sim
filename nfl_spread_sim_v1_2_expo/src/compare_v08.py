from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .uncertainty import fit_uncertainty_model, expected_abs_to_sigma

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/spread_model_v07.joblib")
    ap.add_argument("--training-games", default="artifacts/training_games_with_context.parquet")
    ap.add_argument("--predictions", default="artifacts/v07_context_predictions.csv")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    games = pd.read_parquet(args.training_games)
    preds = pd.read_csv(args.predictions)

    features = bundle["features"]

    uncertainty_model, joined = fit_uncertainty_model(
        games,
        preds,
        features,
    )

    joined["expected_abs_error"] = uncertainty_model.predict(joined[features])
    joined["predicted_sigma"] = joined["expected_abs_error"].map(expected_abs_to_sigma)

    # Diagnostics only. This is not a full probabilistic score yet.
    corr = float(
        np.corrcoef(
            joined["expected_abs_error"],
            joined["abs_residual"]
        )[0,1]
    )

    q = pd.qcut(
        joined["expected_abs_error"],
        q=4,
        labels=["low_vol","med_low","med_high","high_vol"],
        duplicates="drop",
    )
    joined["vol_bucket"] = q

    bucket_stats = (
        joined.groupby("vol_bucket", observed=True)
        .agg(
            games=("game_id","size"),
            predicted_abs_error=("expected_abs_error","mean"),
            actual_abs_error=("abs_residual","mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    result = {
        "version": "0.8",
        "games_used": int(len(joined)),
        "uncertainty_error_correlation": corr,
        "volatility_buckets": bucket_stats,
    }

    out = Path("artifacts")
    out.mkdir(exist_ok=True)

    joblib.dump(
        {
            "version": "0.8",
            "uncertainty_model": uncertainty_model,
            "features": features,
            "diagnostics": result,
        },
        out/"uncertainty_model_v08.joblib",
    )

    joined.to_csv(out/"uncertainty_training_v08.csv", index=False)
    (out/"uncertainty_diagnostics_v08.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
