from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path

def run(label, pred, market, outdir):
    cmd = [
        "python","-m","score_model.market_backtest_v16",
        "--market-csv",market,
        "--oos-predictions",pred,
        "--output-dir",outdir,
        "--min-edge","0.00",
    ]
    subprocess.run(cmd, check=True)
    return json.loads((Path(outdir)/"report.json").read_text())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--market-csv",default="data/historical_market.csv")
    ap.add_argument("--baseline-predictions",default="artifacts/dual_market/oos_baseline_v18.csv")
    ap.add_argument("--qb-predictions",default="artifacts/dual_market/oos_qb_v18.csv")
    ap.add_argument("--output",default="artifacts/dual_market/qb_market_comparison_v18.json")
    args=ap.parse_args()

    base = run("baseline", args.baseline_predictions, args.market_csv, "artifacts/market_backtest_v18_baseline")
    qb = run("qb", args.qb_predictions, args.market_csv, "artifacts/market_backtest_v18_qb")

    report = {
        "baseline": base,
        "qb": qb,
        "delta": {
            "spread_roi_qb_minus_base": qb["spread"]["roi"] - base["spread"]["roi"],
            "spread_brier_qb_minus_base": qb["spread"]["brier"] - base["spread"]["brier"],
            "total_roi_qb_minus_base": qb["total"]["roi"] - base["total"]["roi"],
            "total_brier_qb_minus_base": qb["total"]["brier"] - base["total"]["brier"],
        }
    }
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
