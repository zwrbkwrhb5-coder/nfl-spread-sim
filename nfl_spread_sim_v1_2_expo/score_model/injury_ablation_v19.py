from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SPREAD_BASE = [
    "diff_off_epa","diff_def_epa","diff_off_success","diff_def_success",
    "diff_qb_epa_per_dropback","diff_qb_success_rate","diff_qb_cpoe",
    "diff_qb_sack_rate","diff_qb_interception_rate","diff_qb_explosive_pass_rate",
]

TOTAL_BASE = [
    "sum_off_epa","sum_def_epa","sum_off_success","sum_def_success",
    "sum_points_for","sum_points_against",
    "home_pre_adj_off_epa","away_pre_adj_off_epa",
    "home_pre_adj_def_epa","away_pre_adj_def_epa",
    "home_pre_explosive_pass_rate","away_pre_explosive_pass_rate",
    "home_pre_turnover_rate","away_pre_turnover_rate",
    "home_pre_qb_epa_per_dropback","away_pre_qb_epa_per_dropback",
    "home_pre_qb_success_rate","away_pre_qb_success_rate",
    "home_pre_qb_cpoe","away_pre_qb_cpoe",
    "home_pre_qb_sack_rate","away_pre_qb_sack_rate",
    "home_pre_qb_interception_rate","away_pre_qb_interception_rate",
    "home_pre_qb_explosive_pass_rate","away_pre_qb_explosive_pass_rate",
]

SPREAD_INJURY = [
    "diff_total_injury_impact",
    "diff_qb_injury_impact",
    "diff_ol_injury_impact",
    "diff_secondary_injury_impact",
    "diff_skill_injury_impact",
]

TOTAL_INJURY = [
    "sum_total_injury_impact",
    "sum_qb_injury_impact",
    "sum_ol_injury_impact",
    "sum_secondary_injury_impact",
    "sum_skill_injury_impact",
]


def evaluate_common(df, base, extra, target, first_test_season=2022, max_test_season=2024):
    required = base + extra + [target]
    common = df.dropna(subset=required).copy()
    common = common[common["season"] <= max_test_season]

    rows=[]
    for season in sorted(s for s in common.season.unique() if s >= first_test_season):
        tr=common[common.season < season]
        te=common[common.season == season]
        if tr.empty or te.empty:
            continue
        item={"season":int(season),"games":int(len(te))}
        for label,features in [("base",base),("plus_injury",base+extra)]:
            m=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=1.0))])
            m.fit(tr[features],tr[target])
            p=m.predict(te[features])
            item[f"{label}_mae"]=float(mean_absolute_error(te[target],p))
            item[f"{label}_rmse"]=float(np.sqrt(mean_squared_error(te[target],p)))
        rows.append(item)

    r=pd.DataFrame(rows)
    w=r.games/r.games.sum()
    wm=lambda c: float((r[c]*w).sum())

    return {
        "games":int(r.games.sum()),
        "base_mae":wm("base_mae"),
        "injury_mae":wm("plus_injury_mae"),
        "delta_mae":wm("plus_injury_mae")-wm("base_mae"),
        "base_rmse":wm("base_rmse"),
        "injury_rmse":wm("plus_injury_rmse"),
        "delta_rmse":wm("plus_injury_rmse")-wm("base_rmse"),
        "by_season":rows,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--training-csv",default="artifacts/dual_market/training_games_qb_injury_v19.csv")
    ap.add_argument("--output",default="artifacts/dual_market/injury_ablation_v19.json")
    ap.add_argument("--first-test-season",type=int,default=2022)
    ap.add_argument("--max-test-season",type=int,default=2024)
    args=ap.parse_args()

    df=pd.read_csv(args.training_csv)

    report={
        "spread":evaluate_common(
            df,SPREAD_BASE,SPREAD_INJURY,"actual_margin",
            args.first_test_season,args.max_test_season
        ),
        "total":evaluate_common(
            df,TOTAL_BASE,TOTAL_INJURY,"actual_total",
            args.first_test_season,args.max_test_season
        ),
        "notes":{
            "common_sample":True,
            "qb_baseline":True,
            "max_test_season":args.max_test_season,
            "reason_for_2024_default":"public nflverse injury coverage is not assumed beyond 2024",
        }
    }
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
