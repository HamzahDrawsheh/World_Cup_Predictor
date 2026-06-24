"""Predict every World Cup 2026 group-stage match."""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import clean_data, load_results
from src.elo import EloRating
from src.features import build_feature_matrix
from src.model import build_feature_lookup, load_models, predict_match
from src.utils import WC2026_GROUPS, normalize_team_name

PROCESSED_DIR = ROOT / "data" / "processed"


def refresh_features() -> pd.DataFrame:
    """Rebuild features.csv from the latest downloaded results."""
    raw = load_results()
    cleaned = clean_data(raw)
    elo = EloRating()
    elo.compute_all_ratings(cleaned)
    return build_feature_matrix(cleaned, elo)


def load_or_build_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return refresh_features()


def predict_all_group_matches(neutral: bool = True) -> pd.DataFrame:
    """Return predictions for all 72 group-stage matches (12 groups × 6 fixtures)."""
    models = load_models()
    (
        xgb_model,
        poisson_home,
        poisson_away,
        scaler,
        imputer,
        poisson_imputer,
        poisson_scaler,
        feature_columns,
    ) = models

    features_df = load_or_build_features()
    lookup = build_feature_lookup(features_df)

    rows = []
    for group, teams in WC2026_GROUPS.items():
        for home, away in combinations(teams, 2):
            result = predict_match(
                home_team=home,
                away_team=away,
                feature_lookup=lookup,
                xgb_model=xgb_model,
                poisson_home=poisson_home,
                poisson_away=poisson_away,
                scaler=scaler,
                imputer=imputer,
                poisson_imputer=poisson_imputer,
                poisson_scaler=poisson_scaler,
                feature_columns=feature_columns,
                neutral=neutral,
                is_world_cup=True,
            )
            rows.append(
                {
                    "group": group,
                    "home_team": normalize_team_name(home),
                    "away_team": normalize_team_name(away),
                    "prob_home_win_pct": round(100 * result["prob_home_win"], 1),
                    "prob_draw_pct": round(100 * result["prob_draw"], 1),
                    "prob_away_win_pct": round(100 * result["prob_away_win"], 1),
                    "expected_score": f"{result['expected_home_goals']:.1f}-{result['expected_away_goals']:.1f}",
                    "most_likely_score": result["most_likely_score"],
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = predict_all_group_matches()
    out = ROOT / "data" / "processed" / "wc2026_group_predictions.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} group-stage predictions to {out}\n")
    print(df.to_string(index=False))
