"""Elo rating system for international football teams."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import clean_data, load_results
from src.utils import normalize_team_name

PROCESSED_DIR = ROOT / "data" / "processed"


class EloRating:
    """Chronological Elo ratings with tournament-specific K-factors."""

    def __init__(self, initial_rating: float = 1500) -> None:
        self.initial_rating = initial_rating
        self.ratings: dict[str, float] = {}
        self.history: list[dict] = []

    def _get_k(self, row: pd.Series) -> float:
        if row.get("is_world_cup", False) or row.get("tournament") == "FIFA World Cup":
            return 40.0
        if row.get("is_competitive", False):
            return 30.0
        return 20.0

    def _expected_home(self, home_elo: float, away_elo: float, neutral: bool) -> float:
        home_adv = 0.0 if neutral else 100.0
        exponent = (away_elo - (home_elo + home_adv)) / 400.0
        return 1.0 / (1.0 + 10.0**exponent)

    def compute_all_ratings(self, df: pd.DataFrame) -> None:
        """Iterate matches in date order and update ratings."""
        self.ratings = {}
        self.history = []
        sorted_df = df.sort_values("date").reset_index(drop=True)

        for _, row in sorted_df.iterrows():
            home = normalize_team_name(row["home_team"])
            away = normalize_team_name(row["away_team"])
            neutral = bool(row["neutral"])

            home_elo_before = self.ratings.get(home, self.initial_rating)
            away_elo_before = self.ratings.get(away, self.initial_rating)

            expected_home = self._expected_home(home_elo_before, away_elo_before, neutral)
            expected_away = 1.0 - expected_home

            if row["result"] == "H":
                actual_home, actual_away = 1.0, 0.0
            elif row["result"] == "D":
                actual_home, actual_away = 0.5, 0.5
            else:
                actual_home, actual_away = 0.0, 1.0

            k = self._get_k(row)
            home_elo_after = home_elo_before + k * (actual_home - expected_home)
            away_elo_after = away_elo_before + k * (actual_away - expected_away)

            self.ratings[home] = home_elo_after
            self.ratings[away] = away_elo_after

            self.history.append(
                {
                    "date": row["date"],
                    "home_team": home,
                    "away_team": away,
                    "home_elo_before": home_elo_before,
                    "away_elo_before": away_elo_before,
                    "home_elo_after": home_elo_after,
                    "away_elo_after": away_elo_after,
                }
            )

    def get_rating(self, team: str, before_date) -> float:
        """Return a team's Elo rating immediately before a given date."""
        team = normalize_team_name(team)
        before_ts = pd.Timestamp(before_date)
        rating = self.initial_rating
        for snap in self.history:
            if snap["date"] >= before_ts:
                break
            if snap["home_team"] == team:
                rating = snap["home_elo_after"]
            elif snap["away_team"] == team:
                rating = snap["away_elo_after"]
        return rating

    def get_latest_ratings(self) -> dict[str, float]:
        """Return current Elo ratings for all teams."""
        return dict(self.ratings)

    def get_ratings_history_df(self) -> pd.DataFrame:
        """Return match-level Elo history as a DataFrame."""
        return pd.DataFrame(self.history)


def add_elo_to_df(df: pd.DataFrame, elo_obj: EloRating) -> pd.DataFrame:
    """Attach pre- and post-match Elo columns to the results dataframe."""
    history_df = elo_obj.get_ratings_history_df()
    enriched = df.merge(
        history_df,
        on=["date", "home_team", "away_team"],
        how="left",
    )
    enriched["elo_diff"] = enriched["home_elo_before"] - enriched["away_elo_before"]
    enriched["home_elo"] = enriched["home_elo_before"]
    enriched["away_elo"] = enriched["away_elo_before"]
    return enriched


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_results()
    cleaned = clean_data(raw)
    elo = EloRating()
    elo.compute_all_ratings(cleaned)
    history = elo.get_ratings_history_df()
    history.to_csv(PROCESSED_DIR / "elo_history.csv", index=False)

    top20 = sorted(elo.get_latest_ratings().items(), key=lambda x: x[1], reverse=True)[:20]
    print("Top 20 teams by current Elo:")
    for team, rating in top20:
        print(f"  {team}: {rating:.1f}")
