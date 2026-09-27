from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .qb_tuned_v18 import (
    SPREAD_BASE, SPREAD_QB, TOTAL_BASE, TOTAL_QB, shrink_qb_features
)


def fit_predict_walk_forward(df, spread_features, total_features, first_test_season, alpha):
    rows=[]
    for season in sorted(s for s in df.season.unique() if s >= first_test_season):
        tr = df[df.season < season].copy()
        te = df[df.season == season].copy()

        spread_cols = spread_features + ["actual_margin"]
        total_cols = total_features + ["actual_total"]

        te_idx = te.dropna(subset=spread_features + total_features + ["actual_margin","actual_total"]).index
        te2 = te.loc[te_idx].copy()
        if te2.empty:
            continue

        tr_s = tr.dropna(subset=spread_cols)
        tr_t = tr.dropna(subset=total_cols)

        sm = Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=alpha))])
        tm = Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=alpha))])

        sm.fit(tr_s[spread_features], tr_s["actual_margin"])
        tm.fit(tr_t[total_features], tr_t["actual_total"])

        pred_margin = sm.predict(te2[spread_features])
        pred_total = tm.predict(te2[total_features])

        pred_home = (pred_total + pred_margin)/2
        pred_away = (pred_total - pred_margin)/2

        for i,(_,r) in enumerate(te2.iterrows()):
            rows.append({
                "game_id":r["game_id"],
                "season":int(r["season"]),
                "week":int(r["week"]),
                "home_score":r["home_score"],
                "away_score":r["away_score"],
                "pred_margin":float(pred_margin[i]),
                "pred_total":float(pred_total[i]),
                "pred_home_score":float(pred_home[i]),
                "pred_away_score":float(pred_away[i]),
                "actual_margin":float(r["actual_margin"]),
                "actual_total":float(r["actual_total"]),
            })
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--training-csv",default="artifacts/dual_market/training_games_qb_v17.csv")
    ap.add_argument("--shrinkage-dropbacks",type=float,default=300.0)
    ap.add_argument("--first-test-season",type=int,default=2022)
    ap.add_argument("--alpha",type=float,default=1.0)
    ap.add_argument("--baseline-output",default="artifacts/dual_market/oos_baseline_v18.csv")
    ap.add_argument("--qb-output",default="artifacts/dual_market/oos_qb_v18.csv")
    args=ap.parse_args()

    df=pd.read_csv(args.training_csv)
    d=shrink_qb_features(df,args.shrinkage_dropbacks)

    base=fit_predict_walk_forward(
        d, SPREAD_BASE, TOTAL_BASE, args.first_test_season, args.alpha
    )
    qb=fit_predict_walk_forward(
        d, SPREAD_BASE+SPREAD_QB, TOTAL_BASE+TOTAL_QB,
        args.first_test_season,args.alpha
    )

    Path(args.baseline_output).parent.mkdir(parents=True,exist_ok=True)
    base.to_csv(args.baseline_output,index=False)
    qb.to_csv(args.qb_output,index=False)

    print(f"Saved baseline OOS: {args.baseline_output} ({len(base):,} games)")
    print(f"Saved QB OOS:       {args.qb_output} ({len(qb):,} games)")


if __name__=="__main__":
    main()
