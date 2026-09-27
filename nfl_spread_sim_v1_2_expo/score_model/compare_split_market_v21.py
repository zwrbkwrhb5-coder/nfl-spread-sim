from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_backtest(predictions, market_csv, output_dir):
    cmd = [
        "python",
        "-m",
        "score_model.market_backtest_v16",
        "--market-csv",
        market_csv,
        "--oos-predictions",
        predictions,
        "--output-dir",
        output_dir,
        "--min-edge",
        "0.00",
    ]
    subprocess.run(cmd, check=True)

    p = Path(output_dir) / "report.json"
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-csv", default="data/historical_market.csv")
    ap.add_argument(
        "--qb-predictions",
        default="artifacts/dual_market/oos_qb_common_v21.csv",
    )
    ap.add_argument(
        "--split-predictions",
        default="artifacts/dual_market/oos_split_injury_v21.csv",
    )
    ap.add_argument(
        "--output",
        default="artifacts/dual_market/split_injury_market_v21.json",
    )
    args = ap.parse_args()

    qb = run_backtest(
        args.qb_predictions,
        args.market_csv,
        "artifacts/market_backtest_v21_qb",
    )
    split = run_backtest(
        args.split_predictions,
        args.market_csv,
        "artifacts/market_backtest_v21_split",
    )

    report = {
        "qb_baseline": qb,
        "split_injury": split,
        "delta": {
            "spread_roi_split_minus_qb":
                split["spread"]["roi"] - qb["spread"]["roi"],
            "spread_brier_split_minus_qb":
                split["spread"]["brier"] - qb["spread"]["brier"],
            "total_roi_split_minus_qb":
                split["total"]["roi"] - qb["total"]["roi"],
            "total_brier_split_minus_qb":
                split["total"]["brier"] - qb["total"]["brier"],
        },
        "configuration": {
            "spread": "QB baseline + player-value injuries",
            "total": "QB baseline + basic injury burden",
            "common_sample_required": True,
            "week_safe": True,
        },
    }

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
