from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NFLVERSE_PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp/play_by_play_{season}.parquet"
)

SPREAD_FEATURES = [
    "diff_off_epa","diff_def_epa","diff_off_success","diff_def_success",
    "diff_qb_epa_per_dropback","diff_qb_success_rate","diff_qb_cpoe",
    "diff_qb_sack_rate","diff_qb_interception_rate","diff_qb_explosive_pass_rate",
]

TOTAL_FEATURES = [
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

def american_break_even(odds):
    odds=float(odds)
    return (-odds)/((-odds)+100.0) if odds < 0 else 100.0/(odds+100.0)

def load_recent_pbp(seasons):
    frames=[]
    cols = [
        "game_id","season","week","posteam","defteam","passer_player_id",
        "passer_player_name","pass_attempt","sack","qb_scramble","epa","success",
        "cpoe","interception","yards_gained","turnover","home_team","away_team",
        "home_score","away_score"
    ]
    for s in seasons:
        print(f"Loading PBP {s}...")
        x=pd.read_parquet(NFLVERSE_PBP_URL.format(season=s))
        keep=[c for c in cols if c in x.columns]
        frames.append(x[keep].copy())
    return pd.concat(frames,ignore_index=True)

def build_team_state(pbp):
    d=pbp.copy()
    # one row per team-game, offense and defense
    plays=d[d["posteam"].notna() & d["defteam"].notna()].copy()
    plays["success"]=pd.to_numeric(plays.get("success",np.nan),errors="coerce")
    plays["epa"]=pd.to_numeric(plays.get("epa",np.nan),errors="coerce")
    plays["pass_attempt"]=pd.to_numeric(plays.get("pass_attempt",0),errors="coerce").fillna(0)
    plays["sack"]=pd.to_numeric(plays.get("sack",0),errors="coerce").fillna(0)
    if "turnover" in plays.columns:
        plays["turnover"] = pd.to_numeric(
            plays["turnover"], errors="coerce"
        ).fillna(0)
    else:
        interception = pd.to_numeric(
            plays["interception"], errors="coerce"
        ).fillna(0) if "interception" in plays.columns else pd.Series(0, index=plays.index)

        fumble_lost = pd.to_numeric(
            plays["fumble_lost"], errors="coerce"
        ).fillna(0) if "fumble_lost" in plays.columns else pd.Series(0, index=plays.index)

        plays["turnover"] = (
            (interception > 0) | (fumble_lost > 0)
        ).astype(int)

    plays["yards_gained"]=pd.to_numeric(
        plays["yards_gained"], errors="coerce"
    ) if "yards_gained" in plays.columns else pd.Series(np.nan, index=plays.index)
    plays["explosive_pass"]=((plays["pass_attempt"]>0)&(plays["yards_gained"]>=20)).astype(float)

    off=plays.groupby(["game_id","season","week","posteam"],as_index=False).agg(
        off_epa=("epa","mean"),
        off_success=("success","mean"),
        explosive_pass_rate=("explosive_pass","mean"),
        turnover_rate=("turnover","mean"),
    ).rename(columns={"posteam":"team"})

    deff=plays.groupby(["game_id","season","week","defteam"],as_index=False).agg(
        def_epa=("epa","mean"),
        def_success=("success","mean"),
    ).rename(columns={"defteam":"team"})

    tg=off.merge(deff,on=["game_id","season","week","team"],how="outer")

    # points from final game scores
    games=d.groupby(["game_id","season","week"],as_index=False).agg(
        home_team=("home_team","first"),
        away_team=("away_team","first"),
        home_score=("home_score","max"),
        away_score=("away_score","max"),
    )
    home=games[["game_id","season","week","home_team","home_score","away_score"]].rename(
        columns={"home_team":"team","home_score":"points_for","away_score":"points_against"})
    away=games[["game_id","season","week","away_team","away_score","home_score"]].rename(
        columns={"away_team":"team","away_score":"points_for","home_score":"points_against"})
    pts=pd.concat([home,away],ignore_index=True)
    tg=tg.merge(pts,on=["game_id","season","week","team"],how="left")
    return tg.sort_values(["team","season","week","game_id"])

def latest_team_features(team_games, target_season=2026, target_week=3, halflife=6):
    hist=team_games[
        (team_games["season"]<target_season) |
        ((team_games["season"]==target_season)&(team_games["week"]<target_week))
    ].copy()
    rows=[]
    for team,g in hist.groupby("team"):
        g=g.sort_values(["season","week","game_id"])
        def ewm(c):
            s=pd.to_numeric(g[c],errors="coerce")
            return float(s.ewm(halflife=halflife,adjust=False).mean().iloc[-1])
        rows.append({
            "team":team,
            "pre_off_epa":ewm("off_epa"),
            "pre_def_epa":ewm("def_epa"),
            "pre_off_success":ewm("off_success"),
            "pre_def_success":ewm("def_success"),
            "pre_explosive_pass_rate":ewm("explosive_pass_rate"),
            "pre_turnover_rate":ewm("turnover_rate"),
            "pre_points_for":ewm("points_for"),
            "pre_points_against":ewm("points_against"),
            # live fallback: use raw EWM as opponent-adjusted proxy
            "pre_adj_off_epa":ewm("off_epa"),
            "pre_adj_def_epa":ewm("def_epa"),
        })
    return pd.DataFrame(rows)

def build_qb_state(pbp,target_season=2026,target_week=3,halflife=5,shrink=100):
    d=pbp.copy()
    pa=pd.to_numeric(d.get("pass_attempt",0),errors="coerce").fillna(0)
    sack=pd.to_numeric(d.get("sack",0),errors="coerce").fillna(0)
    scr=pd.to_numeric(d.get("qb_scramble",0),errors="coerce").fillna(0)
    d["_db"]=((pa>0)|(sack>0)|(scr>0)).astype(int)
    d=d[(d["_db"]==1)&d["passer_player_id"].notna()&d["posteam"].notna()].copy()
    d["qb_id"]=d["passer_player_id"].astype(str)
    d["epa"]=pd.to_numeric(d.get("epa",np.nan),errors="coerce")
    d["success"]=pd.to_numeric(d.get("success",np.nan),errors="coerce")
    d["cpoe"]=pd.to_numeric(d.get("cpoe",np.nan),errors="coerce")
    d["sack"]=sack.loc[d.index]
    d["interception"]=pd.to_numeric(d.get("interception",0),errors="coerce").fillna(0)
    yd=pd.to_numeric(d.get("yards_gained",np.nan),errors="coerce")
    d["explosive"]=((pa.loc[d.index]>0)&(yd>=20)).astype(float)

    qg=d.groupby(["game_id","season","week","posteam","qb_id"],as_index=False).agg(
        dropbacks=("_db","sum"),
        qb_epa_per_dropback=("epa","mean"),
        qb_success_rate=("success","mean"),
        qb_cpoe=("cpoe","mean"),
        qb_sack_rate=("sack","mean"),
        qb_interception_rate=("interception","mean"),
        qb_explosive_pass_rate=("explosive","mean"),
    )
    # starter = most dropbacks in most recent game for that team
    qg=qg.sort_values(["posteam","season","week","game_id","dropbacks"])
    recent=qg[
        (qg["season"]<target_season) |
        ((qg["season"]==target_season)&(qg["week"]<target_week))
    ].copy()

    out=[]
    metrics=[
        "qb_epa_per_dropback","qb_success_rate","qb_cpoe",
        "qb_sack_rate","qb_interception_rate","qb_explosive_pass_rate"
    ]
    for team,gteam in recent.groupby("posteam"):
        last_game=gteam.sort_values(["season","week","game_id"]).iloc[-1]["game_id"]
        starter=gteam[gteam["game_id"]==last_game].sort_values("dropbacks",ascending=False).iloc[0]["qb_id"]
        q=gteam[gteam["qb_id"]==starter].sort_values(["season","week","game_id"])
        prior_db=float(q["dropbacks"].sum())
        w=prior_db/(prior_db+shrink)
        row={"team":team,"qb_id":starter,"qb_prior_dropbacks":prior_db}
        for m in metrics:
            raw=float(q[m].ewm(halflife=halflife,adjust=False).mean().iloc[-1])
            lg=float(recent[m].mean(skipna=True))
            row[f"pre_{m}"]=lg+w*(raw-lg)
        out.append(row)
    return pd.DataFrame(out)

def build_live_features(market,team_state,qb_state):
    h=team_state.add_prefix("home_").rename(columns={"home_team":"home_team"})
    a=team_state.add_prefix("away_").rename(columns={"away_team":"away_team"})
    hq=qb_state.add_prefix("home_").rename(columns={"home_team":"home_team"})
    aq=qb_state.add_prefix("away_").rename(columns={"away_team":"away_team"})

    x=market.merge(h,on="home_team",how="left").merge(a,on="away_team",how="left")
    x=x.merge(hq,on="home_team",how="left").merge(aq,on="away_team",how="left")

    x["diff_off_epa"]=x["home_pre_off_epa"]-x["away_pre_off_epa"]
    x["diff_def_epa"]=x["home_pre_def_epa"]-x["away_pre_def_epa"]
    x["diff_off_success"]=x["home_pre_off_success"]-x["away_pre_off_success"]
    x["diff_def_success"]=x["home_pre_def_success"]-x["away_pre_def_success"]

    for m in ["qb_epa_per_dropback","qb_success_rate","qb_cpoe","qb_sack_rate",
              "qb_interception_rate","qb_explosive_pass_rate"]:
        x[f"diff_{m}"]=x[f"home_pre_{m}"]-x[f"away_pre_{m}"]

    for m in ["off_epa","def_epa","off_success","def_success","points_for","points_against"]:
        x[f"sum_{m}"]=x[f"home_pre_{m}"]+x[f"away_pre_{m}"]

    return x

def empirical_prob(errors, threshold):
    # P(error > threshold)
    e=np.asarray(errors,float)
    if len(e)==0: return np.nan
    return float(np.mean(e > threshold))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--training-csv",default="artifacts/dual_market/training_games_qb_v17.csv")
    ap.add_argument("--oos-csv",default="artifacts/dual_market/oos_qb_v18_common.csv")
    ap.add_argument("--market-csv",default="live_data/week3_2026_market.csv")
    ap.add_argument("--season",type=int,default=2026)
    ap.add_argument("--week",type=int,default=3)
    ap.add_argument("--output",default="artifacts/live/week3_2026_ranked.csv")
    args=ap.parse_args()

    train=pd.read_csv(args.training_csv)
    market=pd.read_csv(args.market_csv)

    # Current form: 2025 + completed 2026 games.
    pbp=load_recent_pbp([2025,2026])
    team_games=build_team_state(pbp)
    team_state=latest_team_features(team_games,args.season,args.week)
    qb_state=build_qb_state(pbp,args.season,args.week,shrink=100)
    live=build_live_features(market,team_state,qb_state)

    # Train through available historical table.
    tr_s=train.dropna(subset=SPREAD_FEATURES+["actual_margin"])
    tr_t=train.dropna(subset=TOTAL_FEATURES+["actual_total"])

    sm=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=1.0))])
    tm=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=1.0))])
    sm.fit(tr_s[SPREAD_FEATURES],tr_s["actual_margin"])
    tm.fit(tr_t[TOTAL_FEATURES],tr_t["actual_total"])

    missing=live[SPREAD_FEATURES+TOTAL_FEATURES].isna().any(axis=1)
    if missing.any():
        bad=live.loc[missing,["game_id","away_team","home_team"]]
        print("WARNING missing live features; these games will be excluded:")
        print(bad.to_string(index=False))
    live=live.loc[~missing].copy()

    live["model_margin"]=sm.predict(live[SPREAD_FEATURES])
    live["model_total"]=tm.predict(live[TOTAL_FEATURES])
    live["model_home_score"]=(live["model_total"]+live["model_margin"])/2
    live["model_away_score"]=(live["model_total"]-live["model_margin"])/2

    # Historical residual distributions from OOS QB predictions.
    oos=pd.read_csv(args.oos_csv)
    mres=(oos["actual_margin"]-oos["pred_margin"]).dropna().to_numpy(float)
    tres=(oos["actual_total"]-oos["pred_total"]).dropna().to_numpy(float)

    rows=[]
    for _,r in live.iterrows():
        # home cover iff actual_margin + home_spread > 0.
        # actual_margin = model_margin + residual
        hthr=-(r["model_margin"]+r["home_spread"])
        p_home=empirical_prob(mres,hthr)
        p_away=1-p_home

        tthr=r["total_line"]-r["model_total"]
        p_over=empirical_prob(tres,tthr)
        p_under=1-p_over

        options=[
            ("spread_home",r["home_team"],p_home,r["home_spread_odds"]),
            ("spread_away",r["away_team"],p_away,r["away_spread_odds"]),
            ("total_over","OVER",p_over,r["over_odds"]),
            ("total_under","UNDER",p_under,r["under_odds"]),
        ]
        for market_type,side,p,odds in options:
            be=american_break_even(odds)
            rows.append({
                "game_id":r["game_id"],
                "away_team":r["away_team"],
                "home_team":r["home_team"],
                "market_type":market_type,
                "side":side,
                "probability":p,
                "odds":odds,
                "break_even":be,
                "edge":p-be,
                "home_spread":r["home_spread"],
                "market_total":r["total_line"],
                "model_margin":r["model_margin"],
                "model_total":r["model_total"],
                "model_home_score":r["model_home_score"],
                "model_away_score":r["model_away_score"],
            })

    ranked=pd.DataFrame(rows).sort_values("edge",ascending=False).reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    ranked.to_csv(args.output,index=False)

    print("\nTOP 10 MODEL OPPORTUNITIES")
    print(ranked.head(10)[[
        "game_id","market_type","side","probability","break_even","edge",
        "model_margin","home_spread","model_total","market_total"
    ]].to_string(index=False))
    print(f"\nSaved {args.output}")

if __name__=="__main__":
    main()
