"""Train and serve match prediction models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import clean_data, load_results
from src.elo import EloRating
from src.features import build_feature_matrix
from src.utils import normalize_team_name, safe_divide

MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"

EXCLUDE_COLS = {
    "outcome",
    "home_goals",
    "away_goals",
    "date",
    "home_team",
    "away_team",
    "tournament",
    "city",
    "country",
    "result",
}

POISSON_FEATURES = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "recent_goals_scored_H",
    "recent_goals_conceded_A",
    "recent_goals_scored_A",
    "recent_goals_conceded_H",
    "is_neutral",
    "is_competitive",
]

TEAM_STAT_SUFFIXES = [
    "recent_wins",
    "recent_draws",
    "recent_losses",
    "recent_goals_scored",
    "recent_goals_conceded",
    "recent_goal_diff",
    "recent_points",
    "comp_wins",
    "comp_goal_diff",
]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def _time_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(df["date"])
    train = df[dates < "2022-01-01"]
    val = df[(dates >= "2022-01-01") & (dates < "2024-01-01")]
    test = df[dates >= "2024-01-01"]
    return train, val, test


def _brier_score_multiclass(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    scores = {}
    for cls in range(proba.shape[1]):
        y_bin = (y_true == cls).astype(int)
        scores[f"class_{cls}"] = float(np.mean((proba[:, cls] - y_bin) ** 2))
    scores["mean"] = float(np.mean(list(scores.values())))
    return scores


def _extract_team_stats(row: pd.Series, team: str) -> dict[str, float]:
    """Pull unprefixed team stats from a feature row."""
    if normalize_team_name(row["home_team"]) == normalize_team_name(team):
        prefix = "_H"
    else:
        prefix = "_A"
    stats: dict[str, float] = {}
    for base in TEAM_STAT_SUFFIXES:
        col = f"{base}{prefix}"
        if col in row.index:
            stats[base] = float(row[col]) if pd.notna(row[col]) else np.nan
    stats["elo"] = float(row["home_elo"] if prefix == "_H" else row["away_elo"])
    return stats


def build_feature_lookup(features_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Build latest known per-team stats from the feature matrix."""
    lookup: dict[str, dict[str, float]] = {}
    ordered = features_df.sort_values("date")
    for _, row in ordered.iterrows():
        lookup[normalize_team_name(row["home_team"])] = _extract_team_stats(row, row["home_team"])
        lookup[normalize_team_name(row["away_team"])] = _extract_team_stats(row, row["away_team"])
    return lookup


def _build_match_features(
    home_team: str,
    away_team: str,
    feature_lookup: dict[str, dict[str, float]],
    neutral: bool,
    is_world_cup: bool,
) -> dict[str, float]:
    home_team = normalize_team_name(home_team)
    away_team = normalize_team_name(away_team)
    home_stats = feature_lookup.get(home_team, {})
    away_stats = feature_lookup.get(away_team, {})

    features: dict[str, float] = {}
    for base in TEAM_STAT_SUFFIXES:
        features[f"{base}_H"] = home_stats.get(base, np.nan)
        features[f"{base}_A"] = away_stats.get(base, np.nan)

    home_elo = home_stats.get("elo", 1500.0)
    away_elo = away_stats.get("elo", 1500.0)
    features["home_elo"] = home_elo
    features["away_elo"] = away_elo
    features["elo_diff"] = home_elo - away_elo
    features["is_neutral"] = float(neutral)
    features["is_world_cup"] = float(is_world_cup)
    features["is_competitive"] = 1.0
    features["h2h_wins_H"] = 0.0
    features["h2h_wins_A"] = 0.0
    features["h2h_draws"] = 0.0
    features["h2h_goals_H_avg"] = 0.0
    features["h2h_goals_A_avg"] = 0.0
    features["h2h_total"] = 0.0
    return features


def predict_match(
    home_team: str,
    away_team: str,
    feature_lookup: dict,
    xgb_model,
    poisson_home,
    poisson_away,
    scaler,
    imputer,
    poisson_imputer,
    poisson_scaler,
    feature_columns: list[str],
    neutral: bool = True,
    is_world_cup: bool = True,
) -> dict:
    """Predict outcome probabilities and expected goals for a matchup."""
    raw = _build_match_features(home_team, away_team, feature_lookup, neutral, is_world_cup)
    frame = pd.DataFrame([raw])[feature_columns]
    X = imputer.transform(frame)
    X_scaled = scaler.transform(X)

    proba = xgb_model.predict_proba(X_scaled)[0]
    poisson_frame = pd.DataFrame([raw])[POISSON_FEATURES]
    poisson_imputed = poisson_imputer.transform(poisson_frame)
    poisson_scaled = poisson_scaler.transform(poisson_imputed)
    exp_home = float(np.clip(poisson_home.predict(poisson_scaled)[0], 0.05, 5.0))
    exp_away = float(np.clip(poisson_away.predict(poisson_scaled)[0], 0.05, 5.0))

    max_goals = 6
    best_score = "0-0"
    best_prob = -1.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            prob = poisson.pmf(hg, exp_home) * poisson.pmf(ag, exp_away)
            if prob > best_prob:
                best_prob = prob
                best_score = f"{hg}-{ag}"

    return {
        "home_team": normalize_team_name(home_team),
        "away_team": normalize_team_name(away_team),
        "prob_home_win": float(proba[0]),
        "prob_draw": float(proba[1]),
        "prob_away_win": float(proba[2]),
        "expected_home_goals": exp_home,
        "expected_away_goals": exp_away,
        "most_likely_score": best_score,
    }


def load_models() -> tuple:
    """Load trained models and preprocessors from models/."""
    xgb_model = joblib.load(MODELS_DIR / "xgb_outcome.pkl")
    poisson_home = joblib.load(MODELS_DIR / "poisson_home.pkl")
    poisson_away = joblib.load(MODELS_DIR / "poisson_away.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    imputer = joblib.load(MODELS_DIR / "imputer.pkl")
    poisson_imputer = joblib.load(MODELS_DIR / "poisson_imputer.pkl")
    poisson_scaler = joblib.load(MODELS_DIR / "poisson_scaler.pkl")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
    return (
        xgb_model,
        poisson_home,
        poisson_away,
        scaler,
        imputer,
        poisson_imputer,
        poisson_scaler,
        feature_columns,
    )


def retrain_poisson_models(features_df: pd.DataFrame | None = None) -> None:
    """Retrain Poisson goal models only (fast — no XGBoost/Optuna)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if features_df is None:
        features_df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])

    train_df, _, test_df = _time_splits(features_df)

    poisson_imputer = SimpleImputer(strategy="median")
    poisson_scaler = StandardScaler()
    X_train = poisson_imputer.fit_transform(train_df[POISSON_FEATURES])
    X_train_scaled = poisson_scaler.fit_transform(X_train)
    X_test_scaled = poisson_scaler.transform(poisson_imputer.transform(test_df[POISSON_FEATURES]))

    poisson_home = PoissonRegressor(max_iter=1000, alpha=0.01)
    poisson_away = PoissonRegressor(max_iter=1000, alpha=0.01)
    poisson_home.fit(X_train_scaled, train_df["home_goals"])
    poisson_away.fit(X_train_scaled, train_df["away_goals"])

    home_pred = poisson_home.predict(X_test_scaled)
    away_pred = poisson_away.predict(X_test_scaled)
    print(f"Poisson home MAE: {np.mean(np.abs(home_pred - test_df['home_goals'])):.4f}")
    print(f"Poisson away MAE: {np.mean(np.abs(away_pred - test_df['away_goals'])):.4f}")
    print(f"Poisson home coef (non-zero): {np.count_nonzero(poisson_home.coef_)}/{len(poisson_home.coef_)}")

    joblib.dump(poisson_home, MODELS_DIR / "poisson_home.pkl")
    joblib.dump(poisson_away, MODELS_DIR / "poisson_away.pkl")
    joblib.dump(poisson_imputer, MODELS_DIR / "poisson_imputer.pkl")
    joblib.dump(poisson_scaler, MODELS_DIR / "poisson_scaler.pkl")


def train_models(features_df: pd.DataFrame | None = None) -> None:
    """Train, evaluate, and persist all models."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if features_df is None:
        path = PROCESSED_DIR / "features.csv"
        if path.exists():
            features_df = pd.read_csv(path, parse_dates=["date"])
        else:
            raw = load_results()
            cleaned = clean_data(raw)
            elo = EloRating()
            elo.compute_all_ratings(cleaned)
            features_df = build_feature_matrix(cleaned, elo)

    feature_cols = _feature_columns(features_df)
    train_df, val_df, test_df = _time_splits(features_df)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(train_df[feature_cols])
    X_val = imputer.transform(val_df[feature_cols])
    X_test = imputer.transform(test_df[feature_cols])

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    y_train = train_df["outcome"].astype(int).values
    y_val = val_df["outcome"].astype(int).values
    y_test = test_df["outcome"].astype(int).values

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "verbosity": 0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False,
        )
        preds = model.predict_proba(X_val_scaled)
        return log_loss(y_val, preds, labels=[0, 1, 2])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    best_params = study.best_params
    best_params.update(
        {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "early_stopping_rounds": 50,
            "verbosity": 0,
        }
    )
    xgb_model = xgb.XGBClassifier(**best_params)
    xgb_model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False,
    )

    test_proba = xgb_model.predict_proba(X_test_scaled)
    test_pred = xgb_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, test_pred)
    ll = log_loss(y_test, test_proba, labels=[0, 1, 2])
    brier = _brier_score_multiclass(y_test, test_proba)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1, 2]).tolist()

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test log loss: {ll:.4f}")
    print(f"Brier scores: {brier}")
    print(f"Confusion matrix:\n{np.array(cm)}")

    poisson_imputer = SimpleImputer(strategy="median")
    poisson_scaler = StandardScaler()
    X_poisson_train = poisson_imputer.fit_transform(train_df[POISSON_FEATURES])
    X_poisson_train_scaled = poisson_scaler.fit_transform(X_poisson_train)
    X_poisson_test_scaled = poisson_scaler.transform(
        poisson_imputer.transform(test_df[POISSON_FEATURES])
    )

    poisson_home = PoissonRegressor(max_iter=1000, alpha=0.01)
    poisson_away = PoissonRegressor(max_iter=1000, alpha=0.01)
    poisson_home.fit(X_poisson_train_scaled, train_df["home_goals"])
    poisson_away.fit(X_poisson_train_scaled, train_df["away_goals"])

    home_pred = poisson_home.predict(X_poisson_test_scaled)
    away_pred = poisson_away.predict(X_poisson_test_scaled)
    home_mae = float(np.mean(np.abs(home_pred - test_df["home_goals"])))
    away_mae = float(np.mean(np.abs(away_pred - test_df["away_goals"])))
    home_rmse = float(np.sqrt(np.mean((home_pred - test_df["home_goals"]) ** 2)))
    away_rmse = float(np.sqrt(np.mean((away_pred - test_df["away_goals"]) ** 2)))
    print(f"Poisson home MAE: {home_mae:.4f}, RMSE: {home_rmse:.4f}")
    print(f"Poisson away MAE: {away_mae:.4f}, RMSE: {away_rmse:.4f}")

    importances = dict(zip(feature_cols, xgb_model.feature_importances_.tolist()))
    top_importances = dict(
        sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
    )

    eval_metrics = {
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": brier,
        "confusion_matrix": cm,
        "feature_importances": top_importances,
        "poisson_home_mae": home_mae,
        "poisson_away_mae": away_mae,
    }

    joblib.dump(xgb_model, MODELS_DIR / "xgb_outcome.pkl")
    joblib.dump(poisson_home, MODELS_DIR / "poisson_home.pkl")
    joblib.dump(poisson_away, MODELS_DIR / "poisson_away.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(imputer, MODELS_DIR / "imputer.pkl")
    joblib.dump(poisson_imputer, MODELS_DIR / "poisson_imputer.pkl")
    joblib.dump(poisson_scaler, MODELS_DIR / "poisson_scaler.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.pkl")
    with open(MODELS_DIR / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)


if __name__ == "__main__":
    train_models()
