from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline import SpreadPipeline
from src.slate import build_top5_from_game_results

st.set_page_config(
    page_title="NFL Spread Simulator",
    page_icon="🏈",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
div[data-testid="stDataFrame"] {border-radius: 10px;}
.small-muted {opacity: .70; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

st.title("NFL Spread Simulator")
st.caption("Personal spread model • 1,000,000 simulations per selected game")

DATA_DIR = Path("app_data")
DATA_DIR.mkdir(exist_ok=True)

games_path = DATA_DIR / "games.csv"
market_path = DATA_DIR / "market.csv"
features_dir = DATA_DIR / "features"
features_dir.mkdir(exist_ok=True)

def load_games():
    if games_path.exists():
        return pd.read_csv(games_path)
    return pd.DataFrame(columns=["game_id","week","kickoff","away_team","home_team","feature_csv"])

def load_market():
    if market_path.exists():
        return pd.read_csv(market_path)
    return pd.DataFrame()

games = load_games()
market = load_market()

with st.sidebar:
    st.subheader("Model")
    st.write("Version 1.1")
    st.write("Spread only")
    st.write("1,000,000 final simulations")
    st.divider()

    if games.empty:
        st.warning("No games loaded yet.")
    else:
        weeks = sorted(games["week"].dropna().unique().tolist())
        selected_week = st.selectbox("Week", weeks, index=0)
        games_view = games[games["week"].eq(selected_week)].copy()
        labels = (
            games_view["away_team"].astype(str) + " @ " +
            games_view["home_team"].astype(str)
        ).tolist()
        selected_label = st.selectbox("Game", labels)

if games.empty:
    st.info("""
Add current games to `app_data/games.csv`, market odds to
`app_data/market.csv`, and one matchup feature CSV per game under
`app_data/features/`.

A ready-to-edit template is included with the project.
""")
    st.stop()

row = games_view.iloc[labels.index(selected_label)]
game_id = str(row["game_id"])
home = str(row["home_team"])
away = str(row["away_team"])
feature_csv = Path(str(row["feature_csv"]))

st.subheader(f"{away} @ {home}")
kickoff = str(row.get("kickoff", ""))
if kickoff and kickoff != "nan":
    st.caption(kickoff)

required_model = Path("artifacts/spread_model_v10.joblib")
if not required_model.exists():
    st.error(
        "The trained margin model artifact is not present yet. "
        "Train the v1.0/v1.1 model first, then reopen the app."
    )
    st.stop()

if not feature_csv.exists():
    st.error(f"Missing matchup feature file: {feature_csv}")
    st.stop()

if market.empty:
    st.error("No sportsbook market data loaded in app_data/market.csv.")
    st.stop()

X = pd.read_csv(feature_csv)
if len(X) != 1:
    st.error("Each matchup feature file must contain exactly one row.")
    st.stop()

try:
    pipe = SpreadPipeline()
    result = pipe.analyze_books(
        X=X,
        market=market,
        game_id=game_id,
        home_team=home,
        away_team=away,
        scan_sims=250_000,
        final_sims=1_000_000,
    )
except Exception as e:
    st.error(f"Could not run simulation: {e}")
    st.stop()

best = result["best_opportunity"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model projection", f"{result['projected_winner']} {result['projected_margin_abs']:.1f}")
c2.metric("Best spread", f"{best['side_team']} {float(best['spread']):+g}")
c3.metric("Best odds", f"{int(best['american_odds']):+d}")
c4.metric("Sportsbook", str(best["sportsbook"]))

c5, c6, c7, c8 = st.columns(4)
c5.metric("Cover probability", f"{100*float(best['full_sim_cover_probability']):.1f}%")
c6.metric("Break-even", f"{100*float(best['break_even_probability']):.1f}%")
c7.metric("Probability edge", f"{100*float(best['probability_edge']):+.1f}%")
c8.metric("Projected differential", f"{float(best['projected_differential']):+.1f} pts")

st.divider()

left, right = st.columns([1.15, 1])

with left:
    st.markdown("### 1,000,000 Simulation Summary")
    st.write(
        f"Projected home margin: **{result['projected_home_margin']:+.2f}**  \n"
        f"Matchup volatility (sigma): **{result['target_sigma']:.2f}**"
    )

    sim_rows = pd.DataFrame([
        ["Selected side", best["side_team"]],
        ["Sportsbook", best["sportsbook"]],
        ["Spread", f"{float(best['spread']):+g}"],
        ["American odds", f"{int(best['american_odds']):+d}"],
        ["Full simulations", f"{int(best['full_simulations']):,}"],
        ["Cover probability", f"{100*float(best['full_sim_cover_probability']):.2f}%"],
        ["Model-vs-market diff", f"{float(best['projected_differential']):+.2f} pts"],
    ], columns=["Metric","Result"])
    st.dataframe(sim_rows, hide_index=True, use_container_width=True)

with right:
    st.markdown("### Sportsbook Comparison")
    books = pd.DataFrame(result["all_book_edges"]).copy()
    show_cols = [
        "sportsbook","side_team","spread","american_odds",
        "sim_cover_probability","break_even_probability",
        "probability_edge","projected_differential"
    ]
    show_cols = [c for c in show_cols if c in books.columns]
    if not books.empty:
        display = books[show_cols].copy()
        for c in ["sim_cover_probability","break_even_probability","probability_edge"]:
            if c in display.columns:
                display[c] = display[c].map(lambda x: f"{100*float(x):.1f}%")
        st.dataframe(display, hide_index=True, use_container_width=True)

st.divider()
st.markdown("### Top 5 Slate Edges")

# Run all loaded games if their feature files are available.
slate_results = []
for _, gr in games_view.iterrows():
    fp = Path(str(gr["feature_csv"]))
    if not fp.exists():
        continue
    try:
        gx = pd.read_csv(fp)
        if len(gx) != 1:
            continue
        slate_results.append(
            pipe.analyze_books(
                X=gx,
                market=market,
                game_id=str(gr["game_id"]),
                home_team=str(gr["home_team"]),
                away_team=str(gr["away_team"]),
                scan_sims=100_000,
                final_sims=250_000,
            )
        )
    except Exception:
        continue

top5 = build_top5_from_game_results(slate_results)

if top5.empty:
    st.info("Top 5 will populate once multiple games have valid features and market odds.")
else:
    cols = [
        "away_team","home_team","side_team","sportsbook","spread",
        "american_odds","sim_cover_probability","probability_edge",
        "projected_differential"
    ]
    cols = [c for c in cols if c in top5.columns]
    display = top5[cols].copy()
    for c in ["sim_cover_probability","probability_edge"]:
        if c in display.columns:
            display[c] = display[c].map(lambda x: f"{100*float(x):.1f}%")
    st.dataframe(display, hide_index=True, use_container_width=True)

st.caption(
    "Model probabilities are experimental and should be evaluated with honest "
    "walk-forward backtests before relying on them."
)
