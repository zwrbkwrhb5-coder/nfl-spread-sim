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


def load_groups(path):
    return json.loads(Path(path).read_text())


def evaluate(df, features, target, first_test_season, alpha):
    by_season=[]
    for season in sorted(s for s in df.season.unique() if s>=first_test_season):
        tr=df[df.season<season].dropna(subset=features+[target])
        te=df[df.season==season].dropna(subset=features+[target])
        if tr.empty or te.empty: continue
        model=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=alpha))])
        model.fit(tr[features],tr[target])
        p=model.predict(te[features])
        by_season.append({
            "season":int(season),"games":int(len(te)),
            "mae":float(mean_absolute_error(te[target],p)),
            "rmse":float(np.sqrt(mean_squared_error(te[target],p))),
        })
    if not by_season: return None
    x=pd.DataFrame(by_season)
    w=x.games/x.games.sum()
    return {
        "games":int(x.games.sum()),
        "mae":float((x.mae*w).sum()),
        "rmse":float((x.rmse*w).sum()),
        "by_season":by_season,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--training-csv",default="artifacts/dual_market/training_games.csv")
    ap.add_argument("--groups-json",default="config/feature_groups_v16.json")
    ap.add_argument("--first-test-season",type=int,default=2022)
    ap.add_argument("--alpha",type=float,default=1.0)
    ap.add_argument("--output",default="artifacts/feature_ablation_v16.json")
    args=ap.parse_args()

    df=pd.read_csv(args.training_csv)
    groups=load_groups(args.groups_json)

    target_map={"margin":"actual_margin","total":"actual_total"}
    results={}

    base=groups.get("base",[])
    for target_name,target_col in target_map.items():
        target_results=[]
        base_result=evaluate(df,base,target_col,args.first_test_season,args.alpha)
        target_results.append({"name":"base","features":base,"result":base_result})

        for name,cols in groups.items():
            if name=="base": continue
            feats=base+cols
            r=evaluate(df,feats,target_col,args.first_test_season,args.alpha)
            row={"name":name,"features":cols,"result":r}
            if r and base_result:
                row["delta_mae_vs_base"]=r["mae"]-base_result["mae"]
                row["delta_rmse_vs_base"]=r["rmse"]-base_result["rmse"]
            target_results.append(row)
        results[target_name]=target_results

    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
