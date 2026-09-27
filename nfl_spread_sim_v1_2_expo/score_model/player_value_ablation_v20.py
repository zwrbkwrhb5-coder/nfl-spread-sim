from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

QB_SPREAD_BASE = [
    "diff_off_epa","diff_def_epa","diff_off_success","diff_def_success",
    "diff_qb_epa_per_dropback","diff_qb_success_rate","diff_qb_cpoe",
    "diff_qb_sack_rate","diff_qb_interception_rate","diff_qb_explosive_pass_rate",
]

QB_TOTAL_BASE = [
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

BASIC_SPREAD_INJ = [
    "diff_total_injury_impact",
    "diff_qb_injury_impact",
    "diff_ol_injury_impact",
    "diff_secondary_injury_impact",
    "diff_skill_injury_impact",
]

BASIC_TOTAL_INJ = [
    "sum_total_injury_impact",
    "sum_qb_injury_impact",
    "sum_ol_injury_impact",
    "sum_secondary_injury_impact",
    "sum_skill_injury_impact",
]

PV_SPREAD_INJ = [
    "diff_pv_total_injury_impact",
    "diff_pv_qb_injury_impact",
    "diff_pv_ol_injury_impact",
    "diff_pv_secondary_injury_impact",
    "diff_pv_skill_injury_impact",
]

PV_TOTAL_INJ = [
    "sum_pv_total_injury_impact",
    "sum_pv_qb_injury_impact",
    "sum_pv_ol_injury_impact",
    "sum_pv_secondary_injury_impact",
    "sum_pv_skill_injury_impact",
]


def fit_eval(train, test, features, target):
    m=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=1.0))])
    m.fit(train[features],train[target])
    p=m.predict(test[features])
    return (
        float(mean_absolute_error(test[target],p)),
        float(np.sqrt(mean_squared_error(test[target],p))),
    )


def compare(df, base, basic, pv, target, first_test_season=2022, max_test_season=2024):
    required = base + basic + pv + [target]
    common = df.dropna(subset=required).copy()
    common = common[common.season <= max_test_season]

    rows=[]
    for season in sorted(s for s in common.season.unique() if s >= first_test_season):
        tr=common[common.season < season]
        te=common[common.season == season]
        if tr.empty or te.empty:
            continue

        base_mae,base_rmse=fit_eval(tr,te,base,target)
        basic_mae,basic_rmse=fit_eval(tr,te,base+basic,target)
        pv_mae,pv_rmse=fit_eval(tr,te,base+pv,target)

        rows.append({
            "season":int(season),"games":int(len(te)),
            "base_mae":base_mae,"base_rmse":base_rmse,
            "basic_mae":basic_mae,"basic_rmse":basic_rmse,
            "pv_mae":pv_mae,"pv_rmse":pv_rmse,
        })

    r=pd.DataFrame(rows)
    w=r.games/r.games.sum()
    wm=lambda c: float((r[c]*w).sum())

    return {
        "games":int(r.games.sum()),
        "base_mae":wm("base_mae"),
        "basic_mae":wm("basic_mae"),
        "pv_mae":wm("pv_mae"),
        "pv_delta_vs_base_mae":wm("pv_mae")-wm("base_mae"),
        "pv_delta_vs_basic_mae":wm("pv_mae")-wm("basic_mae"),
        "base_rmse":wm("base_rmse"),
        "basic_rmse":wm("basic_rmse"),
        "pv_rmse":wm("pv_rmse"),
        "pv_delta_vs_base_rmse":wm("pv_rmse")-wm("base_rmse"),
        "pv_delta_vs_basic_rmse":wm("pv_rmse")-wm("basic_rmse"),
        "by_season":rows,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--training-csv",default="artifacts/dual_market/training_games_qb_injury_pv_v20.csv")
    ap.add_argument("--output",default="artifacts/dual_market/player_value_ablation_v20.json")
    ap.add_argument("--first-test-season",type=int,default=2022)
    ap.add_argument("--max-test-season",type=int,default=2024)
    args=ap.parse_args()

    df=pd.read_csv(args.training_csv)

    report={
        "spread":compare(
            df,QB_SPREAD_BASE,BASIC_SPREAD_INJ,PV_SPREAD_INJ,
            "actual_margin",args.first_test_season,args.max_test_season
        ),
        "total":compare(
            df,QB_TOTAL_BASE,BASIC_TOTAL_INJ,PV_TOTAL_INJ,
            "actual_total",args.first_test_season,args.max_test_season
        ),
        "notes":{
            "common_sample":True,
            "comparison":"QB baseline vs QB+basic injuries vs QB+player-value injuries",
            "max_test_season":args.max_test_season,
        }
    }

    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
