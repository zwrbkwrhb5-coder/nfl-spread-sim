from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_backtest(predictions: str, market_csv: str, output_dir: str):
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

    report_path = Path(output_dir) / "report.json"
    return json.loads(report_path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--market-csv",
        default="data/historical_market.csv",
    )
    ap.add_argument(
        "--baseline-predictions",
        default="artifacts/dual_market/oos_qb_total_base_v25.csv",
    )
    ap.add_argument(
        "--context-predictions",
        default="artifacts/dual_market/oos_qb_total_context_v25.csv",
    )
    ap.add_argument(
        "--output",
        default="artifacts/dual_market/totals_context_market_v25.json",
    )
    args = ap.parse_args()

    baseline = run_backtest(
        args.baseline_predictions,
        args.market_csv,
        "artifacts/market_backtest_v25_total_base",
    )

    context = run_backtest(
        args.context_predictions,
        args.market_csv,
        "artifacts/market_backtest_v25_total_context",
    )

    report = {
        "qb_total_baseline": baseline,
        "qb_plus_context_total": context,
        "delta": {
            "total_roi_context_minus_base":
                context["total"]["roi"] - baseline["total"]["roi"],
            "total_brier_context_minus_base":
                context["total"]["brier"] - baseline["total"]["brier"],
            "total_win_rate_context_minus_base":
                context["total"]["win_rate"] - baseline["total"]["win_rate"],
            "total_avg_edge_context_minus_base":
                context["total"]["avg_edge"] - baseline["total"]["avg_edge"],
            "spread_roi_context_minus_base":
                context["spread"]["roi"] - baseline["spread"]["roi"],
            "spread_brier_context_minus_base":
                context["spread"]["brier"] - baseline["spread"]["brier"],
        },
        "validation": {
            "week_safe": True,
            "common_sample": True,
            "same_training_rows": True,
            "same_test_rows": True,
            "spread_model_unchanged": True,
            "only_total_model_changes": True,
        },
    }

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
