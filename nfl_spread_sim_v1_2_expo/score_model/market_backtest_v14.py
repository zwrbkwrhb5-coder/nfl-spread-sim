from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

def break_even(odds: float) -> float:
    odds = float(odds)
    return 100/(odds+100) if odds > 0 else (-odds)/((-odds)+100)

def win_profit(odds: float) -> float:
    odds = float(odds)
    return odds/100 if odds > 0 else 100/(-odds)

def empirical_prob(proj: float, residuals: np.ndarray, threshold: float, gt: bool):
    sims = proj + residuals
    push = np.isclose(sims, threshold)
    win = sims > threshold if gt else sims < threshold
    return float(win.mean()), float(push.mean())

def actual_spread_grade(actual_margin: float, home_spread: float):
    v = actual_margin + home_spread
    if np.isclose(v, 0): return "push"
    return "home" if v > 0 else "away"

def actual_total_grade(actual_total: float, line: float):
    v = actual_total - line
    if np.isclose(v, 0): return "push"
    return "over" if v > 0 else "under"

def calibrate_walk_forward(rows: pd.DataFrame, min_rows: int = 100) -> pd.DataFrame:
    out = rows.copy()
    out["calibrated_probability"] = out["raw_probability"]
    for market in out["market"].unique():
        md = out[out["market"].eq(market)]
        for season in sorted(md["season"].unique()):
            tr = md[(md["season"] < season) & (~md["is_push"])]
            te_idx = md[md["season"].eq(season)].index
            if len(tr) < min_rows or tr["won"].nunique() < 2:
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(tr["raw_probability"], tr["won"])
            out.loc[te_idx,"calibrated_probability"] = iso.predict(
                out.loc[te_idx,"raw_probability"]
            )
    out["probability_edge"] = out["calibrated_probability"] - out["break_even_probability"]
    return out

def build_rows(market: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    p = pred.copy()
    if "actual_margin" not in p:
        p["actual_margin"] = p["actual_home_score"] - p["actual_away_score"]
    if "actual_total" not in p:
        p["actual_total"] = p["actual_home_score"] + p["actual_away_score"]

    margin_resid = (p["actual_margin"] - p["pred_margin"]).dropna().to_numpy(float)
    total_resid = (p["actual_total"] - p["pred_total"]).dropna().to_numpy(float)

    m = market.merge(
        p[["game_id","season","pred_margin","pred_total"]],
        on=["game_id","season"], how="inner", validate="one_to_one"
    )
    rows = []
    for _, r in m.iterrows():
        am = float(r["home_score"] - r["away_score"])
        at = float(r["home_score"] + r["away_score"])
        sgrade = actual_spread_grade(am, float(r["home_spread"]))
        tgrade = actual_total_grade(at, float(r["total_line"]))

        # Home spread
        hp, hpushp = empirical_prob(
            float(r["pred_margin"]), margin_resid, -float(r["home_spread"]), True
        )
        rows.append(dict(
            game_id=r.game_id, season=int(r.season), market="spread",
            selection=str(r.home_team), side="home",
            line=float(r.home_spread), odds=float(r.home_spread_odds),
            raw_probability=hp, push_probability=hpushp,
            break_even_probability=break_even(r.home_spread_odds),
            line_edge_points=float(r.pred_margin + r.home_spread),
            won=int(sgrade=="home"), is_push=bool(sgrade=="push"),
            price_imputed=bool(r.get("home_spread_odds_imputed", False)),
        ))

        # Away spread — direct complement from same residual bank
        ap, apushp = empirical_prob(
            float(r["pred_margin"]), margin_resid, -float(r["home_spread"]), False
        )
        rows.append(dict(
            game_id=r.game_id, season=int(r.season), market="spread",
            selection=str(r.away_team), side="away",
            line=float(r.away_spread), odds=float(r.away_spread_odds),
            raw_probability=ap, push_probability=apushp,
            break_even_probability=break_even(r.away_spread_odds),
            line_edge_points=float(-(r.pred_margin + r.home_spread)),
            won=int(sgrade=="away"), is_push=bool(sgrade=="push"),
            price_imputed=bool(r.get("away_spread_odds_imputed", False)),
        ))

        # Over
        op, opushp = empirical_prob(
            float(r["pred_total"]), total_resid, float(r["total_line"]), True
        )
        rows.append(dict(
            game_id=r.game_id, season=int(r.season), market="total",
            selection="OVER", side="over",
            line=float(r.total_line), odds=float(r.over_odds),
            raw_probability=op, push_probability=opushp,
            break_even_probability=break_even(r.over_odds),
            line_edge_points=float(r.pred_total-r.total_line),
            won=int(tgrade=="over"), is_push=bool(tgrade=="push"),
            price_imputed=bool(r.get("over_odds_imputed", False)),
        ))

        # Under
        up, upushp = empirical_prob(
            float(r["pred_total"]), total_resid, float(r["total_line"]), False
        )
        rows.append(dict(
            game_id=r.game_id, season=int(r.season), market="total",
            selection="UNDER", side="under",
            line=float(r.total_line), odds=float(r.under_odds),
            raw_probability=up, push_probability=upushp,
            break_even_probability=break_even(r.under_odds),
            line_edge_points=float(r.total_line-r.pred_total),
            won=int(tgrade=="under"), is_push=bool(tgrade=="push"),
            price_imputed=bool(r.get("under_odds_imputed", False)),
        ))
    return pd.DataFrame(rows)

def select_best(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.sort_values(
            ["game_id","market","probability_edge","line_edge_points"],
            ascending=[True,True,False,False]
        )
        .drop_duplicates(["game_id","market"])
        .reset_index(drop=True)
    )

def add_profit(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["profit_units"] = np.where(
        d["is_push"], 0.0,
        np.where(d["won"].eq(1), d["odds"].map(win_profit), -1.0)
    )
    return d

def summary(d: pd.DataFrame):
    settled=d[~d.is_push]
    return dict(
        bets=int(len(d)), settled=int(len(settled)), pushes=int(d.is_push.sum()),
        win_rate=float(settled.won.mean()) if len(settled) else None,
        units=float(d.profit_units.sum()),
        roi=float(d.profit_units.sum()/len(settled)) if len(settled) else None,
        avg_edge=float(d.probability_edge.mean()) if len(d) else None,
        imputed_price_rate=float(d.price_imputed.mean()) if len(d) else None,
        brier=float(brier_score_loss(settled.won, settled.calibrated_probability))
          if len(settled) else None,
    )

def edge_buckets(df):
    bins=[-np.inf,0,.01,.02,.03,.05,.075,.10,np.inf]
    labels=["<=0%","0-1%","1-2%","2-3%","3-5%","5-7.5%","7.5-10%","10%+"]
    d=df.copy()
    d["edge_bucket"]=pd.cut(d.probability_edge,bins=bins,labels=labels,right=False)
    out=[]
    for (market,b),g in d.groupby(["market","edge_bucket"],observed=True):
        s=g[~g.is_push]
        out.append(dict(
            market=market, edge_bucket=str(b), bets=len(g), settled=len(s),
            win_rate=float(s.won.mean()) if len(s) else np.nan,
            units=float(g.profit_units.sum()),
            roi=float(g.profit_units.sum()/len(s)) if len(s) else np.nan,
            avg_edge=float(g.probability_edge.mean()),
            avg_line_edge=float(g.line_edge_points.mean()),
        ))
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--market-csv",default="data/historical_market.csv")
    ap.add_argument("--oos-predictions",default="artifacts/dual_market/oos_predictions.csv")
    ap.add_argument("--output-dir",default="artifacts/market_backtest_v14")
    ap.add_argument("--min-edge",type=float,default=0.0)
    args=ap.parse_args()

    market=pd.read_csv(args.market_csv)
    pred=pd.read_csv(args.oos_predictions)
    rows=build_rows(market,pred)
    rows=calibrate_walk_forward(rows)
    rows=add_profit(rows)
    best=select_best(rows)
    bettable=best[best.probability_edge>=args.min_edge].copy()

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows.to_csv(out/"all_sides.csv",index=False)
    best.to_csv(out/"best_spread_and_total_per_game.csv",index=False)
    bettable.to_csv(out/"bettable.csv",index=False)
    edge_buckets(best).to_csv(out/"edge_buckets.csv",index=False)

    report={
        "matched_games":int(best.game_id.nunique()),
        "selected_bets":int(len(bettable)),
        "min_edge":args.min_edge,
        "spread":summary(bettable[bettable.market.eq("spread")]),
        "total":summary(bettable[bettable.market.eq("total")]),
        "important_note":"Closing lines are used only as evaluation market lines. Do not treat them as pregame model features unless they were available at the prediction timestamp."
    }
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
