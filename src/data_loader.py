"""Download and clean international football match results."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import normalize_team_name

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

DATA_URLS = {
    "results.csv": (
        "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    ),
    # Downloaded for future use — not used in the current pipeline.
    "shootouts.csv": (
        "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
    ),
    # Downloaded for future use — not used in the current pipeline.
    "goalscorers.csv": (
        "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
    ),
}


def download_data(raw_dir: Path | None = None, force: bool = False) -> None:
    """Download CSV files to data/raw/, skipping files that already exist unless force=True."""
    raw_dir = raw_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in DATA_URLS.items():
        dest = raw_dir / filename
        if dest.exists() and not force:
            print(f"Already exists, skipping: {dest}")
            continue
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            dest.write_bytes(response.content)
            print(f"Downloaded: {dest}")
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def load_results(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load results.csv with appropriate dtypes."""
    raw_dir = raw_dir or RAW_DIR
    path = raw_dir / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run download_data() first.")

    df = pd.read_csv(
        path,
        parse_dates=["date"],
        dtype={
            "home_team": "string",
            "away_team": "string",
            "tournament": "string",
            "city": "string",
            "country": "string",
        },
    )
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").astype("Int64")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").astype("Int64")
    df["neutral"] = df["neutral"].astype(bool)
    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the raw results dataframe."""
    cleaned = df.dropna(subset=["home_score", "away_score"]).copy()
    cleaned = cleaned[cleaned["date"] >= pd.Timestamp("1990-01-01")]

    cleaned["home_score"] = cleaned["home_score"].astype(int)
    cleaned["away_score"] = cleaned["away_score"].astype(int)
    cleaned["goal_diff"] = cleaned["home_score"] - cleaned["away_score"]

    cleaned["result"] = "D"
    cleaned.loc[cleaned["goal_diff"] > 0, "result"] = "H"
    cleaned.loc[cleaned["goal_diff"] < 0, "result"] = "A"

    cleaned["is_world_cup"] = cleaned["tournament"] == "FIFA World Cup"
    cleaned["is_competitive"] = ~cleaned["tournament"].str.contains(
        "Friendly", case=False, na=False
    )

    cleaned["outcome"] = 1
    cleaned.loc[cleaned["result"] == "H", "outcome"] = 0
    cleaned.loc[cleaned["result"] == "A", "outcome"] = 2

    cleaned = cleaned.sort_values("date").reset_index(drop=True)
    return cleaned


def get_team_matches(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return all matches involving a team from that team's perspective."""
    team = normalize_team_name(team)
    home = df[df["home_team"] == team].copy()
    away = df[df["away_team"] == team].copy()

    home["is_home"] = True
    home["goals_for"] = home["home_score"]
    home["goals_against"] = home["away_score"]
    home["opponent"] = home["away_team"]

    away["is_home"] = False
    away["goals_for"] = away["away_score"]
    away["goals_against"] = away["home_score"]
    away["opponent"] = away["home_team"]

    matches = pd.concat([home, away], ignore_index=True)
    matches["team_result"] = "D"
    matches.loc[matches["goals_for"] > matches["goals_against"], "team_result"] = "W"
    matches.loc[matches["goals_for"] < matches["goals_against"], "team_result"] = "L"
    matches = matches.sort_values("date").reset_index(drop=True)
    return matches


if __name__ == "__main__":
    download_data()
    results = load_results()
    cleaned = clean_data(results)
    print(f"Shape: {cleaned.shape}")
    print(cleaned.head())
