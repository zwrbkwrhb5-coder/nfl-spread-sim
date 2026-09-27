# Run the Personal App

## 1. Install

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## 2. Train / place model artifacts

The app expects:

- `artifacts/spread_model_v10.joblib`

Optionally:

- `artifacts/uncertainty_model_v08.joblib`

These are produced by the model training / comparison pipeline.

## 3. Add your weekly data

Edit:

- `app_data/games.csv`
- `app_data/market.csv`

Create one matchup feature CSV per game under:

- `app_data/features/`

## 4. Start the app

```bash
streamlit run app.py
```

Then open the local address Streamlit prints in the terminal.

## User flow

1. Choose week
2. Choose NFL game
3. View projected margin
4. View best sportsbook spread and price
5. View cover probability
6. View projected differential
7. View sportsbook comparison
8. View Top 5 slate edges
