from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

from .pipeline import SpreadPipeline
from .slate import build_top5_from_game_results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="CSV: game_id,home_team,away_team,feature_csv")
    ap.add_argument("--market", required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    market = pd.read_csv(args.market)
    pipe = SpreadPipeline()

    results = []
    for _, r in manifest.iterrows():
        X = pd.read_csv(r["feature_csv"])
        if len(X) != 1:
            raise ValueError(f"{r['feature_csv']} must have exactly one row.")

        results.append(
            pipe.analyze_books(
                X=X,
                market=market,
                game_id=r["game_id"],
                home_team=r["home_team"],
                away_team=r["away_team"],
                scan_sims=250_000,
                final_sims=1_000_000,
            )
        )

    top5 = build_top5_from_game_results(results)

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    top5.to_csv(out/"top5_v1_0.csv", index=False)
    (out/"slate_results_v1_0.json").write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )

    print("TOP 5")
    print(top5.to_string(index=False))

if __name__ == "__main__":
    main()
