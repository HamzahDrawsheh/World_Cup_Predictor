"""Orchestrate the full World Cup 2026 prediction pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import PROCESSED_DIR, RAW_DIR, clean_data, download_data, load_results
from src.elo import EloRating, add_elo_to_df
from src.features import build_feature_matrix
from src.model import train_models
from src.simulator import TournamentSimulator
from src.model import build_feature_lookup, load_models

MODELS_DIR = ROOT / "models"
LOG = logging.getLogger("run_pipeline")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _run_step(name: str, func, force: bool, skip_check) -> str:
    start = time.time()
    LOG.info("Starting step: %s", name)
    try:
        if not force and skip_check():
            duration = time.time() - start
            LOG.info(
                "Step %s | SKIPPED | duration=%.2fs",
                name,
                duration,
            )
            return "SKIPPED"
        func()
        duration = time.time() - start
        LOG.info("Step %s | OK | duration=%.2fs", name, duration)
        return "OK"
    except Exception:
        duration = time.time() - start
        LOG.exception("Step %s | ERROR | duration=%.2fs", name, duration)
        return "ERROR"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the WC 2026 predictor pipeline.")
    parser.add_argument("--force", action="store_true", help="Re-run all steps.")
    args = parser.parse_args()
    force = args.force

    _configure_logging()
    LOG.info("Pipeline started (force=%s)", force)

    _run_step(
        "download_data",
        download_data,
        force,
        lambda: RAW_DIR.joinpath("results.csv").exists()
        and RAW_DIR.joinpath("results.csv").stat().st_size > 1_000_000,
    )

    def compute_elo() -> None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cleaned = clean_data(load_results())
        elo = EloRating()
        elo.compute_all_ratings(cleaned)
        elo.get_ratings_history_df().to_csv(PROCESSED_DIR / "elo_history.csv", index=False)

    _run_step(
        "compute_elo",
        compute_elo,
        force,
        lambda: (PROCESSED_DIR / "elo_history.csv").exists(),
    )

    def build_features() -> None:
        cleaned = clean_data(load_results())
        elo = EloRating()
        elo.compute_all_ratings(cleaned)
        build_feature_matrix(cleaned, elo)

    _run_step(
        "build_features",
        build_features,
        force,
        lambda: (PROCESSED_DIR / "features.csv").exists(),
    )

    _run_step(
        "train_models",
        train_models,
        force,
        lambda: (MODELS_DIR / "xgb_outcome.pkl").exists(),
    )

    def smoke_test_simulator() -> None:
        models = load_models()
        import pandas as pd

        features_df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
        lookup = build_feature_lookup(features_df)
        sim = TournamentSimulator(models, lookup)
        results = sim.run_simulation(n_simulations=100)
        LOG.info("Smoke test top 5:\n%s", results.head(5).to_string(index=False))

    _run_step("simulator_smoke_test", smoke_test_simulator, force=True, skip_check=lambda: False)

    LOG.info('Run: streamlit run dashboard/app.py')
    LOG.info("Pipeline finished")


if __name__ == "__main__":
    main()
