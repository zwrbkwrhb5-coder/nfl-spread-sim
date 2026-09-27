from __future__ import annotations
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from .pipeline import SpreadPipeline

app = FastAPI(title="NFL Spread Simulator API", version="1.2")

@app.get("/health")
def health():
    return {"ok": True, "version": "1.2"}

@app.get("/games")
def games():
    p = Path("app_data/games.csv")
    if not p.exists():
        return []
    return pd.read_csv(p).fillna("").to_dict(orient="records")

@app.get("/game/{game_id}")
def game(game_id: str):
    games_path = Path("app_data/games.csv")
    market_path = Path("app_data/market.csv")
    if not games_path.exists() or not market_path.exists():
        raise HTTPException(404, "App data is not loaded.")

    games = pd.read_csv(games_path)
    match = games[games["game_id"].astype(str).eq(str(game_id))]
    if match.empty:
        raise HTTPException(404, "Game not found.")

    r = match.iloc[0]
    feature_path = Path(str(r["feature_csv"]))
    if not feature_path.exists():
        raise HTTPException(404, "Feature vector not found.")

    try:
        pipe = SpreadPipeline()
        result = pipe.analyze_books(
            X=pd.read_csv(feature_path),
            market=pd.read_csv(market_path),
            game_id=str(game_id),
            home_team=str(r["home_team"]),
            away_team=str(r["away_team"]),
            scan_sims=250_000,
            final_sims=1_000_000,
        )
        return result
    except Exception as exc:
        raise HTTPException(500, str(exc))
