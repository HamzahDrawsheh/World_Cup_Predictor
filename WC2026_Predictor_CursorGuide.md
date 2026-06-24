# World Cup 2026 — Match Prediction Model
## Complete Cursor AI Build Guide

> **Data source:** https://raw.githubusercontent.com/martj42/international_results/master/results.csv  
> **Stack:** Python · pandas · XGBoost · scikit-learn · Streamlit  
> **Architecture:** Feature Engineering → XGBoost + Poisson → Monte Carlo Simulator → Dashboard

---

## Known Issues & Notes (Read Before Starting)

| # | Issue | Fix Applied |
|---|---|---|
| 1 | WC 2026 groups — corrected to official FIFA draw | See Step 5 — verified groups A–L |
| 2 | Team names must match the CSV exactly | See Step 5 — name mapping included |
| 3 | `shootouts.csv` & `goalscorers.csv` — downloaded but not used yet | Kept as future enhancement only; Step 1 only uses `results.csv` actively |
| 4 | `utils.py` has no spec | Added Step 0b with its actual contents |
| 5 | `features.py` on 30k+ rows will be slow | Step 3 uses groupby + precomputed histories, not per-row filtering |
| 6 | PyTorch listed in header but never used | Removed from requirements |

---

## Project Structure

```
world_cup_predictor/
├── data/
│   ├── raw/                  ← downloaded CSVs go here
│   └── processed/            ← cleaned & feature-engineered data
├── src/
│   ├── data_loader.py        ← download & clean data
│   ├── features.py           ← build team features
│   ├── elo.py                ← Elo rating system
│   ├── model.py              ← train XGBoost + Poisson models
│   ├── simulator.py          ← Monte Carlo tournament engine
│   └── utils.py              ← shared helper functions
├── notebooks/
│   ├── 01_exploration.ipynb  ← EDA
│   ├── 02_features.ipynb     ← feature analysis
│   └── 03_model_eval.ipynb   ← model evaluation
├── dashboard/
│   └── app.py                ← Streamlit dashboard
├── models/                   ← saved model files (.pkl, .json)
├── requirements.txt
└── README.md
```

---

## Step 0 — Setup

### Prompt for Cursor:

```
Create a requirements.txt for a football match prediction project with these packages:
pandas, numpy, scikit-learn, xgboost, lightgbm, scipy, matplotlib, seaborn,
streamlit, plotly, requests, joblib, tqdm, jupyter, optuna

Also create a README.md explaining the project:
- World Cup 2026 match outcome prediction
- Uses historical international results from 1872 to present
- Predicts win/draw/loss probabilities for any matchup
- Runs Monte Carlo simulation to predict tournament winner odds
```

---

## Step 0b — Utils

### Prompt for Cursor:

```
Create src/utils.py with these shared helper functions used across the project:

1. normalize_team_name(name: str) -> str
   - Strips extra whitespace, title-cases the name
   - Applies a hardcoded mapping dict for known CSV mismatches:
     {
       'Czechia': 'Czech Republic',   # CSV uses 'Czech Republic'
       'Türkiye': 'Turkey',           # CSV uses 'Turkey'
       'IR Iran': 'Iran',             # CSV uses 'Iran'
       'USA': 'United States',        # CSV uses 'United States'
       'Ivory Coast': "Côte d'Ivoire", # normalize both directions
       'Cote d\'Ivoire': "Côte d'Ivoire",
       'Congo DR': 'DR Congo',        # CSV may use 'DR Congo'
       'Curacao': 'Curaçao',
     }
   - Returns the normalized name, or the original if not in mapping

2. get_all_wc2026_teams() -> list[str]
   - Returns the list of all 48 WC 2026 team names as they appear in the CSV
   - (Use the CSV-normalized names from WC2026_GROUPS in simulator.py)

3. points_from_result(goals_for, goals_against) -> int
   - Returns 3 for win, 1 for draw, 0 for loss

4. safe_divide(a, b, default=0.0) -> float
   - Returns a/b, or default if b is 0

Add docstrings to all functions.
```

---

## Step 1 — Data Loader

### Prompt for Cursor:

```
Create src/data_loader.py that does the following:

DATA SOURCES:
- Main CSV (actively used):
  https://raw.githubusercontent.com/martj42/international_results/master/results.csv
- Shootouts CSV (download and save for future use — not used in current pipeline):
  https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv
- Goalscorers CSV (download and save for future use — not used in current pipeline):
  https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv

COLUMNS in results.csv:
date, home_team, away_team, home_score, away_score, tournament, city, country, neutral

REQUIREMENTS:
1. Function download_data() — downloads all 3 CSVs to data/raw/
   - Print which files were downloaded vs already exist (skip if file present)
   - Handle network errors gracefully with a clear error message

2. Function load_results() -> pd.DataFrame — loads results.csv:
   - date as datetime
   - home_score, away_score as Int64 (nullable int to handle any nulls)
   - neutral as bool

3. Function clean_data(df) -> pd.DataFrame:
   - Drop rows where home_score or away_score is null
   - Filter only matches from 1990 onwards (modern football era)
   - Add 'result' column: 'H' = home win, 'A' = away win, 'D' = draw
   - Add 'goal_diff' = home_score - away_score
   - Add 'is_world_cup' = True if tournament == 'FIFA World Cup'
   - Add 'is_competitive' = True if 'Friendly' not in tournament string
   - Add 'outcome' = 0 (home win), 1 (draw), 2 (away win) — for model target

4. Function get_team_matches(df, team: str) -> pd.DataFrame
   - Returns all matches involving 'team' (both home and away)
   - Adds column 'is_home' bool
   - Adds columns 'goals_for' and 'goals_against' from that team's perspective
   - Adds column 'team_result': 'W', 'D', or 'L' from that team's perspective

5. __main__ block: downloads data, loads, cleans, prints shape and df.head()

Use pathlib for all file paths. Import normalize_team_name from src.utils.
```

---

## Step 2 — Elo Rating System

### Prompt for Cursor:

```
Create src/elo.py implementing an Elo rating system for international football.

REQUIREMENTS:

1. Class EloRating:
   __init__(self, initial_rating=1500)
   
   compute_all_ratings(self, df: pd.DataFrame)
   - Takes full results df sorted ascending by date
   - Iterates every match chronologically
   - Stores rating snapshots: self.history = list of dicts
     {date, home_team, away_team, home_elo_before, away_elo_before,
      home_elo_after, away_elo_after}
   - K-factor by tournament type:
       K=40 for 'FIFA World Cup'
       K=30 for any competitive match (is_competitive=True)
       K=20 for friendlies
   - Home advantage: add 100 to home team's rating when computing
     expected score (skip if neutral=True)
   - Expected score formula:
       E_home = 1 / (1 + 10 ** ((away_elo - (home_elo + home_adv)) / 400))
   - Actual score: 1=win, 0.5=draw, 0=loss
   - Update: new_rating = old_rating + K * (actual - expected)

   get_rating(self, team: str, before_date) -> float
   - Returns the team's Elo rating just before before_date
   - Returns initial_rating (1500) if team has no history yet

   get_latest_ratings(self) -> dict[str, float]
   - Returns {team_name: current_elo} for all teams

   get_ratings_history_df(self) -> pd.DataFrame
   - Returns the full history as a DataFrame

2. Function add_elo_to_df(df, elo_obj) -> pd.DataFrame
   - Adds columns to the results df:
     home_elo_before, away_elo_before, home_elo_after, away_elo_after
     elo_diff = home_elo_before - away_elo_before
   - Must use the pre-match rating (before_date = match date)

3. __main__ block:
   - Load and clean data
   - Compute all ratings
   - Print top 20 teams by current Elo with their ratings
```

---

## Step 3 — Feature Engineering

### Prompt for Cursor:

```
Create src/features.py that builds a feature matrix for match prediction.
CRITICAL: Zero data leakage — all features must use only data strictly before each match date.

PERFORMANCE NOTE: The dataset has 30k+ rows. Do NOT filter inside a per-row loop.
Instead, pre-compute each team's full match history as a sorted list/dict, then
use binary search (bisect) or pre-grouped DataFrames to look up historical stats
efficiently. Use tqdm for a progress bar on the outer loop.

IMPORTS:
  pandas, numpy, tqdm, bisect
  from src.data_loader import load_results, clean_data, get_team_matches
  from src.elo import EloRating, add_elo_to_df
  from src.utils import safe_divide

PRE-COMPUTATION (do this once before the main loop):
  - Group df by team (both as home and away) into team_history dict:
    { team_name: sorted DataFrame of all their matches with goals_for, goals_against,
      result (W/D/L), points (3/1/0), is_competitive, date }
  - Compute Elo ratings for all matches

FEATURES TO BUILD (for each match row, using only data before that match date):

  RECENT FORM (last 10 matches, any tournament):
  - recent_wins_H, recent_wins_A
  - recent_draws_H, recent_draws_A
  - recent_losses_H, recent_losses_A
  - recent_goals_scored_H, recent_goals_scored_A   (average)
  - recent_goals_conceded_H, recent_goals_conceded_A (average)
  - recent_goal_diff_H, recent_goal_diff_A          (average)
  - recent_points_H, recent_points_A               (average points per game)

  COMPETITIVE FORM (last 10 competitive matches only, is_competitive=True):
  - comp_wins_H, comp_wins_A
  - comp_goal_diff_H, comp_goal_diff_A

  HEAD TO HEAD (all previous matches between these two specific teams):
  - h2h_wins_H      (times home_team beat away_team)
  - h2h_wins_A      (times away_team beat home_team)
  - h2h_draws
  - h2h_goals_H_avg (avg goals home_team scores vs this specific opponent)
  - h2h_goals_A_avg
  - h2h_total

  ELO (already in df after add_elo_to_df):
  - home_elo, away_elo, elo_diff

  CONTEXT:
  - is_neutral    (int 0/1)
  - is_world_cup  (int 0/1)
  - is_competitive (int 0/1)

  TARGETS (include in output but exclude from model input):
  - outcome  (0=home win, 1=draw, 2=away win)
  - home_goals, away_goals  (for Poisson model)

MAIN FUNCTION:
  build_feature_matrix(df: pd.DataFrame, elo_obj: EloRating) -> pd.DataFrame
  - Returns complete feature DataFrame
  - Saves to data/processed/features.csv
  - Prints shape and null count summary when done

__main__ block: load data → compute elo → build features → save
```

---

## Step 4 — Model Training

### Prompt for Cursor:

```
Create src/model.py for training and saving the prediction models.

IMPORTS: pandas, numpy, sklearn, xgboost, scipy, joblib, optuna
         from src.features import build_feature_matrix

FEATURE COLUMNS = all columns except: outcome, home_goals, away_goals, date,
                  home_team, away_team, tournament, city, country, result

TIME-BASED SPLIT (never random split for time series data):
  - Train:      matches before 2022-01-01
  - Validation: 2022-01-01 to 2023-12-31
  - Test:       2024-01-01 onwards

PREPROCESSING:
  - Impute nulls with column median (teams with <10 historical matches will have nulls)
  - StandardScaler on all numeric features
  - Save imputer + scaler to models/ with joblib

─────────────────────────────────────────
MODEL A: XGBoost Outcome Classifier
─────────────────────────────────────────
Target: outcome (0=home win, 1=draw, 2=away win)

Hyperparameter tuning with Optuna (50 trials), optimizing val log loss:
  n_estimators: 100–1000
  max_depth: 3–10
  learning_rate: 0.01–0.3
  subsample: 0.6–1.0
  colsample_bytree: 0.6–1.0
  min_child_weight: 1–10

XGBClassifier settings:
  objective='multi:softprob'
  num_class=3
  eval_metric='mlogloss'
  early_stopping_rounds=50
  use_label_encoder=False

Evaluate on test set and print:
  - Accuracy
  - Log Loss
  - Brier Score (per class)
  - Confusion matrix

Save to: models/xgb_outcome.pkl

─────────────────────────────────────────
MODEL B: Poisson Goals Regressor
─────────────────────────────────────────
Two separate models:
  - poisson_home → predicts home_goals
  - poisson_away → predicts away_goals

Use sklearn's PoissonRegressor.
Features: home_elo, away_elo, elo_diff,
          recent_goals_scored_H, recent_goals_conceded_A,
          recent_goals_scored_A, recent_goals_conceded_H,
          is_neutral, is_competitive

Evaluate with MAE and RMSE on test set.
Save to: models/poisson_home.pkl, models/poisson_away.pkl

─────────────────────────────────────────
PREDICTION FUNCTION (critical — used by simulator):
─────────────────────────────────────────

def predict_match(home_team: str,
                  away_team: str,
                  feature_lookup: dict,   ← precomputed latest features per team
                  xgb_model,
                  poisson_home,
                  poisson_away,
                  scaler,
                  imputer,
                  neutral: bool = True,
                  is_world_cup: bool = True) -> dict:

  Builds a feature vector for this matchup using feature_lookup (latest known stats).
  Returns:
  {
    'home_team': str,
    'away_team': str,
    'prob_home_win': float,
    'prob_draw': float,
    'prob_away_win': float,
    'expected_home_goals': float,
    'expected_away_goals': float,
    'most_likely_score': str   # e.g. "2-1"
  }

def build_feature_lookup(features_df: pd.DataFrame) -> dict:
  - Builds a dict {team_name: latest_feature_row} from the feature matrix
  - Used to quickly get current team stats for prediction without re-running features

def load_models() -> tuple:
  - Loads and returns (xgb_model, poisson_home, poisson_away, scaler, imputer)
  - All from models/ folder

__main__ block: trains both models, prints evaluation metrics, saves everything.
```

---

## Step 5 — Tournament Simulator

### Prompt for Cursor:

```
Create src/simulator.py — the Monte Carlo World Cup 2026 tournament engine.

IMPORTS: numpy, pandas, random, copy
         from src.model import predict_match, load_models, build_feature_lookup
         from src.utils import normalize_team_name, points_from_result

─────────────────────────────────────────
OFFICIAL WC 2026 GROUPS (verified from FIFA draw, December 5 2025):
─────────────────────────────────────────
WC2026_GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curacao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cabo Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

IMPORTANT — team names must match martj42 CSV exactly. Known mappings needed:
  'Czechia'   → 'Czech Republic'    (CSV spelling)
  'Türkiye'   → 'Turkey'            (CSV spelling)
  'IR Iran'   → 'Iran'              (CSV spelling)
  'Côte d\'Ivoire' → 'Ivory Coast'  (CSV spelling)
  'Cabo Verde' → 'Cape Verde'       (check CSV — may use 'Cape Verde')
  'Curaçao'   → 'Curacao'           (CSV spelling — check)
  'Congo DR'  → 'DR Congo'          (check CSV)
Apply normalize_team_name() from utils before any lookup.

─────────────────────────────────────────
CLASS: TournamentSimulator
─────────────────────────────────────────

__init__(self, models_tuple, feature_lookup):
  - Stores models + feature_lookup for predict_match calls

simulate_match(home_team, away_team, neutral=True, knockout=False) -> dict:
  - Calls predict_match() to get probabilities
  - For group stage (knockout=False):
      randomly samples outcome from [home_win, draw, away_win] with given probs
      also samples scoreline from Poisson(expected_goals) for each team
      returns {'winner': str or None, 'home_goals': int, 'away_goals': int, 'draw': bool}
  - For knockout stage (knockout=True):
      if draw: simulate 30 min extra time (reduce draw prob by 40%, redistribute)
      if still draw: coin flip for penalties (50/50)
      always returns a winner

simulate_group(teams: list[str]) -> pd.DataFrame:
  - Simulates all 6 matches in a group (round-robin)
  - Returns standings DataFrame with columns:
    team, played, wins, draws, losses, goals_for, goals_against, goal_diff, points
  - Sorted by: points DESC → goal_diff DESC → goals_for DESC

get_third_place_qualifiers(all_standings: dict) -> list[str]:
  - Takes dict of {group_letter: standings_df}
  - Collects all 12 third-place teams
  - Returns best 8 by points → goal_diff → goals_for

simulate_knockout_round(teams: list[str]) -> list[str]:
  - Takes list of teams paired as [match1_teamA, match1_teamB, match2_teamA, ...]
  - Simulates each match (knockout=True), returns list of winners

simulate_tournament(groups: dict = WC2026_GROUPS) -> str:
  - Runs one full tournament: group stage → R32 → R16 → QF → SF → Final
  - Returns the name of the winner

run_simulation(n_simulations: int = 10000, groups: dict = WC2026_GROUPS) -> pd.DataFrame:
  - Runs simulate_tournament() n_simulations times
  - Tracks each team's exit round count
  - Returns DataFrame: team, champion_pct, final_pct, semi_pct, quarter_pct,
                        r16_pct, r32_pct, group_exit_pct
  - Sorted by champion_pct DESC

─────────────────────────────────────────
__main__ block:
─────────────────────────────────────────
- Load models and feature lookup
- Run 10,000 simulations
- Print top 20 teams by championship probability
- Print estimated runtime
```

---

## Step 6 — Streamlit Dashboard

### Prompt for Cursor:

```
Create dashboard/app.py — a Streamlit dashboard for the World Cup 2026 predictor.

SETUP:
  import sys, pathlib
  sys.path.append(str(pathlib.Path(__file__).parent.parent))
  — so that src/ imports work when run from dashboard/ folder

Use st.cache_data to cache: model loading, feature loading, data loading
Use st.sidebar for page navigation with st.radio

PAGES:

─────────────────────────────────────────
PAGE 1: Match Predictor
─────────────────────────────────────────
- Two selectboxes: Home Team, Away Team (all 48 WC teams, sorted alphabetically)
- Checkbox: Neutral venue (default checked)
- Button: "Predict"
- Output (3 columns):
    col1: big metric "Home Win" with probability %
    col2: big metric "Draw" with probability %
    col3: big metric "Away Win" with probability %
- Plotly horizontal bar chart of the three probabilities
- Subheader: "Expected score: X – Y" using Poisson expected goals
- Expander "Head-to-head history": table of last 10 matches between the teams
  (load from results.csv filtered for these two teams)

─────────────────────────────────────────
PAGE 2: Tournament Odds
─────────────────────────────────────────
- Slider: simulations (1000–20000, default 5000, step 1000)
  (cap at 20k for dashboard performance; full 10k+ run via __main__)
- Button: "Run Simulation"
- Show st.spinner while running
- Output:
    Plotly treemap of championship % for all 48 teams
    (size=champion_pct, color by confederation)
    Full sortable table: Team | Champion% | Final% | Semi% | Quarter% | Group Exit%

─────────────────────────────────────────
PAGE 3: Team Profile
─────────────────────────────────────────
- Selectbox: choose any team
- Show in columns:
    Current Elo rating (big metric)
    Elo trend: line chart, last 5 years
    Last 10 results: table with opponent, score, W/D/L, tournament
    Avg goals scored vs conceded (last 20 matches) as a line chart (plotly)

─────────────────────────────────────────
PAGE 4: Model Report
─────────────────────────────────────────
- Loads saved evaluation metrics from models/eval_metrics.json
  (save this in Step 4 — model.py should save accuracy, log_loss, brier_score,
   confusion_matrix, and feature_importances as JSON/dict)
- Displays:
    Accuracy, Log Loss, Brier Score as st.metric cards
    Confusion matrix as annotated plotly heatmap
    Feature importance as horizontal bar chart (top 20 features)

Footer on all pages:
"Data: github.com/martj42/international_results (CC0) | Built with Cursor AI + Streamlit"
```

---

## Step 7 — EDA Notebook

### Prompt for Cursor:

```
Create notebooks/01_exploration.ipynb with these sections:

SETUP: Add parent dir to sys.path so src/ imports work from notebooks/

1. Load & clean data via src/data_loader
2. Basic stats: shape, date range, null counts, tournaments list
3. Matches per year chart (1990–2026) — bar chart
4. Top 20 teams by total matches played — horizontal bar
5. Home advantage: win/draw/loss % overall — grouped bar
6. Home advantage by tournament type (World Cup vs Friendly vs Other) — grouped bar
7. Goals per game by decade — line chart
8. Goal difference distribution — histogram with KDE
9. Most common scorelines in World Cup matches — top 15 table
10. Elo ratings: top 10 teams' Elo over time (2000–2026) — multiline chart
11. Feature correlation heatmap (after loading features.csv)
12. Recent form vs outcome: box plot of recent_points_H grouped by outcome

Use matplotlib + seaborn. Add a markdown cell before each plot explaining what to look for.
```

---

## Step 8 — Run Pipeline

### Prompt for Cursor:

```
Create run_pipeline.py at the project root that runs the full pipeline in order.

Use Python's logging module (not print) for all output.
Log: step name, start time, end time, duration, status (OK / SKIPPED / ERROR).

STEPS (with skip logic):
1. Download data
   SKIP if data/raw/results.csv exists and is >1MB

2. Compute Elo ratings
   SKIP if data/processed/elo_history.csv exists

3. Build feature matrix
   SKIP if data/processed/features.csv exists

4. Train models
   SKIP if models/xgb_outcome.pkl exists

5. Test simulator (100 simulations only — just a smoke test)
   Always runs — fast check that nothing is broken

6. Print instructions to launch dashboard:
   "Run: streamlit run dashboard/app.py"
   Never launch it automatically from this script.

Handle exceptions per step: log the error and continue to next step if possible.
Add --force flag (argparse) to re-run all steps even if outputs exist.
```

---

## Important Tips for Using Cursor

### Do this after each step:
1. **Run the code** before moving to the next step
2. **Paste any error** into Cursor chat — it will fix it
3. **Ask Cursor to explain** anything: *"explain what bisect is doing in features.py"*
4. **Use `@filename`** in Cursor chat to reference a specific file

### Cursor shortcuts:
- `Ctrl+L` — open chat
- `Ctrl+K` — inline edit (select code first)
- `Ctrl+I` — Composer (multi-file edits)
- `@filename` — reference a file in your prompt

### If Cursor loses context:
Open a new chat and paste:
> "I'm building a WC 2026 predictor. Project root: world_cup_predictor/. I just finished Step [N]. Now help me with Step [N+1]. Here is my current [filename]: [paste relevant code]."

---

## Build Order Summary

```
Step 0  → requirements.txt, README.md
Step 0b → src/utils.py
Step 1  → src/data_loader.py        (run it: downloads data)
Step 2  → src/elo.py                (run it: prints top 20 Elo ratings)
Step 3  → src/features.py           (run it: builds features.csv — takes ~2 min)
Step 4  → src/model.py              (run it: trains models — takes ~5 min)
Step 5  → src/simulator.py          (run it: prints championship odds)
Step 6  → dashboard/app.py          (run: streamlit run dashboard/app.py)
Step 7  → notebooks/01_exploration  (open in Jupyter — optional but recommended)
Step 8  → run_pipeline.py           (ties everything together)
```

---

## Expected Results

| Metric | Expected Value |
|---|---|
| Model accuracy | ~55–62% (3-class problem; random = 33%) |
| features.py runtime | 1–3 minutes on modern laptop |
| 10k simulation runtime | 3–8 minutes depending on hardware |
| Dashboard cold start | ~5 seconds (cached after first load) |

---

## Free Deployment

```bash
# 1. Push project to GitHub
# 2. Go to https://share.streamlit.io
# 3. Connect repo → set main file: dashboard/app.py
# 4. Add any secrets if needed → Deploy
# Your predictor is live at a public URL
```

---

*Data: github.com/martj42/international_results — CC0 License, free to use commercially*  
*Built with Cursor AI + Python + Streamlit*
