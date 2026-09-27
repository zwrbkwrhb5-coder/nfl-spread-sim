# Expo Go Setup — v1.2

## Mobile UI demo

Install Node.js, then:

```bash
cd mobile
npm install
npx expo start
```

Scan the QR code with Expo Go on your phone.

The mobile app starts in DEMO mode so the interface can be tested before the
trained model and live data feeds are fully automated.

## Python backend

From the project root:

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The FastAPI backend exposes:

- GET /health
- GET /games
- GET /game/{game_id}

The game endpoint runs the existing model pipeline and the final 1,000,000
simulation analysis.

## Architecture

Expo Go
  ↓
FastAPI
  ↓
pregame feature pipeline
  ↓
margin model
  ↓
matchup uncertainty
  ↓
Monte Carlo simulation
  ↓
sportsbook comparison
  ↓
mobile result

## Next connection

The included Expo interface currently uses local demo data. The next engineering
step is to replace `DEMO_GAMES` with fetch calls to the FastAPI endpoints and
then connect automated NFL / odds / injury / weather feeds.
