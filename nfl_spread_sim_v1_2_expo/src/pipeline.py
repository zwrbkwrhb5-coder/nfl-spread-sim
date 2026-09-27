from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .market import validate_market_snapshot, add_market_model_edges, rank_top_edges
from .market_sim import build_side_probability_function
from .uncertainty import expected_abs_to_sigma
from .sim_engine_v08 import simulate_spread_dynamic

@dataclass
class PipelineArtifacts:
    margin_model_path: str = "artifacts/spread_model_v10.joblib"
    uncertainty_model_path: str = "artifacts/uncertainty_model_v08.joblib"
    games_path: str = "artifacts/training_games_with_qb.parquet"
    qb_games_path: str = "artifacts/qb_games.parquet"

class SpreadPipeline:
    """
    End-to-end inference shell.

    This class assumes upstream feature engineering has already produced
    a one-row matchup feature vector matching the saved margin model.

    v1.0 focuses on orchestration:
      margin -> uncertainty -> simulations -> book comparison -> ranking
    """

    def __init__(self, artifacts: PipelineArtifacts = PipelineArtifacts()):
        self.margin_bundle = joblib.load(artifacts.margin_model_path)
        self.uncertainty_bundle = None
        if Path(artifacts.uncertainty_model_path).exists():
            self.uncertainty_bundle = joblib.load(artifacts.uncertainty_model_path)

    @property
    def features(self):
        return self.margin_bundle["features"]

    @property
    def residuals(self):
        return self.margin_bundle["residuals"]

    def predict_margin(self, X: pd.DataFrame) -> float:
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise ValueError(f"Missing model features: {missing[:10]}")
        return float(self.margin_bundle["model"].predict(X[self.features])[0])

    def predict_sigma(self, X: pd.DataFrame) -> float:
        if self.uncertainty_bundle is None:
            return float(np.std(np.asarray(self.residuals), ddof=1))

        model = self.uncertainty_bundle["uncertainty_model"]
        feats = self.uncertainty_bundle["features"]
        missing = [c for c in feats if c not in X.columns]
        if missing:
            # Conservative fallback to global residual std.
            return float(np.std(np.asarray(self.residuals), ddof=1))

        expected_abs = float(model.predict(X[feats])[0])
        return expected_abs_to_sigma(expected_abs)

    def simulate_selected_line(
        self,
        projected_home_margin: float,
        target_sigma: float,
        home_spread: float,
        odds: int = -110,
        n: int = 1_000_000,
        calibrator=None,
    ):
        return simulate_spread_dynamic(
            projected_home_margin=projected_home_margin,
            residuals=self.residuals,
            target_sigma=target_sigma,
            home_spread=home_spread,
            odds=odds,
            n=n,
            calibrator=calibrator,
        )

    def analyze_books(
        self,
        X: pd.DataFrame,
        market: pd.DataFrame,
        game_id: str,
        home_team: str,
        away_team: str,
        scan_sims: int = 250_000,
        final_sims: int = 1_000_000,
        calibrator=None,
    ):
        projected_home_margin = self.predict_margin(X)
        target_sigma = self.predict_sigma(X)

        market = validate_market_snapshot(market)
        current = market[
            market["game_id"].eq(game_id) &
            market["market_phase"].astype(str).str.lower().eq("current")
        ].copy()

        if current.empty:
            raise ValueError(f"No current market rows for game_id={game_id}")

        projected_margin_by_side = {
            home_team: projected_home_margin,
            away_team: -projected_home_margin,
        }

        prob_fn = build_side_probability_function(
            projected_home_margin=projected_home_margin,
            residuals=self.residuals,
            target_sigma=target_sigma,
            home_team=home_team,
            away_team=away_team,
            n=scan_sims,
            calibrator=calibrator,
        )

        edges = add_market_model_edges(
            current,
            projected_margin_by_side=projected_margin_by_side,
            simulated_cover_probability_fn=prob_fn,
            phase="current",
        )

        # Best opportunity in this game.
        ranked = rank_top_edges(edges, top_n=len(edges))
        best = ranked.iloc[0].to_dict()

        # Re-run best line at full 1,000,000 sims.
        best_side = best["side_team"]
        best_spread = float(best["spread"])
        best_odds = int(best["american_odds"])

        if best_side == home_team:
            home_spread = best_spread
            full = self.simulate_selected_line(
                projected_home_margin,
                target_sigma,
                home_spread,
                odds=best_odds,
                n=final_sims,
                calibrator=calibrator,
            )
            full_side_cover = full["calibrated_cover_probability"]
        else:
            home_spread = -best_spread
            full = self.simulate_selected_line(
                projected_home_margin,
                target_sigma,
                home_spread,
                odds=best_odds,
                n=final_sims,
                calibrator=calibrator,
            )
            full_side_cover = (
                1.0
                - full["calibrated_cover_probability"]
                - full["push_probability"]
            )

        best["full_sim_cover_probability"] = float(full_side_cover)
        best["full_simulations"] = int(final_sims)

        return {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "projected_home_margin": projected_home_margin,
            "projected_winner": home_team if projected_home_margin >= 0 else away_team,
            "projected_margin_abs": abs(projected_home_margin),
            "target_sigma": target_sigma,
            "best_opportunity": best,
            "all_book_edges": ranked.to_dict(orient="records"),
        }
