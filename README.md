# World Cup 2026 Match Predictor

Machine learning pipeline that predicts international football match outcomes and simulates the FIFA World Cup 2026 tournament.

## Live demo

Deploy to **[Streamlit Community Cloud](https://share.streamlit.io)** for a public link (no install needed for visitors).

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

## Deploy to Streamlit Cloud (share a public link)

1. Push this repo to GitHub (models are included for cloud hosting).
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub.
3. Click **Create app** → select `HamzahDrawsheh/World_Cup_Predictor`.
4. **Main file:** `streamlit_app.py` (auto-detected if left default).
5. Click **Deploy**.

**If the app crashes:** open **Manage app → Settings** and set **Python version** to **3.11** (XGBoost does not support 3.14 yet).

On first visit, the app downloads match data (~4 MB) and builds features (~30s). After that, predictions load instantly.

**App URL format:** `https://<your-app-name>.streamlit.app`

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
