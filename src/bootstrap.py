"""Ensure data, features, and models exist (local dev + Streamlit Cloud)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

REQUIRED_MODELS = [
    "xgb_outcome.pkl",
    "poisson_home.pkl",
    "poisson_away.pkl",
    "scaler.pkl",
    "imputer.pkl",
    "poisson_imputer.pkl",
    "poisson_scaler.pkl",
    "feature_columns.pkl",
]


def models_available() -> bool:
    """Return True if all trained model files are present."""
    return all((MODELS_DIR / name).exists() for name in REQUIRED_MODELS)


def ensure_artifacts() -> None:
    """
    Download match data and build features when missing.

    Trained models must be committed in models/ for Streamlit Cloud.
    """
    from src.data_loader import clean_data, download_data, load_results
    from src.elo import EloRating
    from src.features import build_feature_matrix

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results_path = RAW_DIR / "results.csv"
    if not results_path.exists() or results_path.stat().st_size < 1_000_000:
        download_data()

    features_path = PROCESSED_DIR / "features.csv"
    if not features_path.exists():
        cleaned = clean_data(load_results())
        elo = EloRating()
        elo.compute_all_ratings(cleaned)
        build_feature_matrix(cleaned, elo)

    if not models_available():
        missing = [n for n in REQUIRED_MODELS if not (MODELS_DIR / n).exists()]
        raise FileNotFoundError(
            "Missing trained models: "
            + ", ".join(missing)
            + ". Run `python run_pipeline.py` locally or commit models/ to the repo."
        )
