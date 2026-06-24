# **World Cup 2026 Match Predictor**

![Screenshot](./Flags.png)

Welcome to the World Cup 2026 Predictor. Pick any two teams to see predicted win, draw, and loss probabilities, expected goals, and head-to-head history. Run thousands of tournament simulations to explore championship odds for all 48 teams. Predictions are based on recent form, Elo ratings, and historical international results—not live odds or insider knowledge—so treat them as data-driven estimates, not guarantees.

## Live demo

## **Visit** [Live Predictor demo](https://worldcuppredictor-zqxw5owmxxdahkx8xbbtjr.streamlit.app/) 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## Features

- **Match predictor** — win / draw / loss probabilities for any two teams
- **Score estimate** — expected goals and most likely scoreline (Poisson model)
- **Tournament simulator** — Monte Carlo championship odds for all 48 teams
- **Streamlit dashboard** — interactive UI with team profiles and model metrics

## Stack

Python · pandas · XGBoost · scikit-learn · Streamlit · Plotly

## Quick start (local)

```bash
git clone https://github.com/HamzahDrawsheh/World_Cup_Predictor.git
cd World_Cup_Predictor

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python run_pipeline.py          # downloads data, trains models (~15 min first run)
```

For local **model training** (includes Optuna, Jupyter, etc.):

```bash
pip install -r requirements-dev.txt
python run_pipeline.py
```

```bash
streamlit run streamlit_app.py  # open http://localhost:8501
```

## Project structure

```
World_Cup_Predictor/
├── streamlit_app.py      # Streamlit Cloud entry point
├── dashboard/app.py      # Dashboard UI
├── src/                  # data loading, Elo, features, models, simulator
├── run_pipeline.py       # full pipeline orchestrator
├── predict_wc_matches.py # predict all 72 group-stage matches
├── models/               # trained models (committed for cloud deploy)
├── data/                 # downloaded at runtime (gitignored)
└── notebooks/            # EDA notebook
```

## Common commands

| Command | Description |
|---------|-------------|
| `python run_pipeline.py` | Run full pipeline (skips completed steps) |
| `python run_pipeline.py --force` | Re-run everything from scratch |
| `python predict_wc_matches.py` | Export all group-stage predictions to CSV |
| `streamlit run streamlit_app.py` | Launch the dashboard locally |

## Refresh data

```bash
python -c "from src.data_loader import download_data; download_data(force=True)"
python -c "from predict_wc_matches import refresh_features; refresh_features()"
```

## Data source

International match results from [martj42/international_results](https://github.com/martj42/international_results) — **CC0 license**, free to use commercially.
