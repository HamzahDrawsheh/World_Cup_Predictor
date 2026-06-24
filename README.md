# World Cup 2026 Match Predictor

Machine learning pipeline that predicts international football match outcomes and simulates the FIFA World Cup 2026 tournament.

## Features

- **Match predictor** — win / draw / loss probabilities for any two teams
- **Score estimate** — expected goals and most likely scoreline (Poisson model)
- **Tournament simulator** — Monte Carlo championship odds for all 48 teams
- **Streamlit dashboard** — interactive UI with team profiles and model metrics

## Stack

Python · pandas · XGBoost · scikit-learn · Streamlit · Plotly

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/world_cup_predictor.git
cd world_cup_predictor

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python run_pipeline.py          # downloads data, trains models (~15 min first run)
streamlit run dashboard/app.py  # open http://localhost:8501
```

## Project structure

```
world_cup_predictor/
├── src/                  # data loading, Elo, features, models, simulator
├── dashboard/app.py      # Streamlit UI
├── run_pipeline.py       # full pipeline orchestrator
├── predict_wc_matches.py # predict all 72 group-stage matches
├── notebooks/            # EDA notebook
├── data/raw/             # downloaded CSVs (gitignored, created by pipeline)
├── data/processed/       # features & Elo history (gitignored)
└── models/               # trained .pkl files (gitignored)
```

## Common commands

| Command | Description |
|---------|-------------|
| `python run_pipeline.py` | Run full pipeline (skips completed steps) |
| `python run_pipeline.py --force` | Re-run everything from scratch |
| `python predict_wc_matches.py` | Export all group-stage predictions to CSV |
| `streamlit run dashboard/app.py` | Launch the dashboard |

## Refresh data

```bash
python -c "from src.data_loader import download_data; download_data(force=True)"
python -c "from predict_wc_matches import refresh_features; refresh_features()"
```

## Data source

International match results from [martj42/international_results](https://github.com/martj42/international_results) — **CC0 license**, free to use commercially.

## Deploy dashboard (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repo → set main file to `dashboard/app.py`
4. Deploy

Note: Streamlit Cloud will need to run `run_pipeline.py` once (or include trained models) before predictions work.
