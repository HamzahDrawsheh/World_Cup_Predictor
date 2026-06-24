"""Feature engineering for match prediction (leakage-free)."""

from __future__ import annotations

import bisect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import clean_data, load_results
from src.elo import EloRating, add_elo_to_df
from src.utils import normalize_team_name, safe_divide

PROCESSED_DIR = ROOT / "data" / "processed"


def _build_team_histories(df: pd.DataFrame) -> dict[str, dict[str, list[Any]]]:
    """Pre-compute per-team chronological match records for bisect lookups."""
    histories: dict[str, dict[str, list[Any]]] = {}

    def ensure(team: str) -> dict[str, list[Any]]:
        if team not in histories:
            histories[team] = {
                "dates": [],
                "goals_for": [],
                "goals_against": [],
                "results": [],
                "points": [],
                "is_competitive": [],
            }
        return histories[team]

    for row in df.itertuples(index=False):
        home = normalize_team_name(row.home_team)
        away = normalize_team_name(row.away_team)
        date = row.date
        is_comp = bool(row.is_competitive)

        home_hist = ensure(home)
        home_hist["dates"].append(date)
        home_hist["goals_for"].append(int(row.home_score))
        home_hist["goals_against"].append(int(row.away_score))
        if row.home_score > row.away_score:
            home_hist["results"].append("W")
            home_hist["points"].append(3)
        elif row.home_score == row.away_score:
            home_hist["results"].append("D")
            home_hist["points"].append(1)
        else:
            home_hist["results"].append("L")
            home_hist["points"].append(0)
        home_hist["is_competitive"].append(is_comp)

        away_hist = ensure(away)
        away_hist["dates"].append(date)
        away_hist["goals_for"].append(int(row.away_score))
        away_hist["goals_against"].append(int(row.home_score))
        if row.away_score > row.home_score:
            away_hist["results"].append("W")
            away_hist["points"].append(3)
        elif row.away_score == row.home_score:
            away_hist["results"].append("D")
            away_hist["points"].append(1)
        else:
            away_hist["results"].append("L")
            away_hist["points"].append(0)
        away_hist["is_competitive"].append(is_comp)

    return histories


def _build_h2h_histories(df: pd.DataFrame) -> dict[frozenset[str], dict[str, list[Any]]]:
    """Pre-compute head-to-head histories keyed by unordered team pair."""
    h2h: dict[frozenset[str], dict[str, list[Any]]] = {}

    def ensure(pair: frozenset[str]) -> dict[str, list[Any]]:
        if pair not in h2h:
            h2h[pair] = {
                "dates": [],
                "home_team": [],
                "away_team": [],
                "home_score": [],
                "away_score": [],
            }
        return h2h[pair]

    for row in df.itertuples(index=False):
        home = normalize_team_name(row.home_team)
        away = normalize_team_name(row.away_team)
        bucket = ensure(frozenset({home, away}))
        bucket["dates"].append(row.date)
        bucket["home_team"].append(home)
        bucket["away_team"].append(away)
        bucket["home_score"].append(int(row.home_score))
        bucket["away_score"].append(int(row.away_score))

    return h2h


def _slice_before(
    history: dict[str, list[Any]], before_date: pd.Timestamp
) -> dict[str, list[Any]]:
    """Return history records strictly before before_date using bisect."""
    idx = bisect.bisect_left(history["dates"], before_date)
    return {key: values[:idx] for key, values in history.items()}


def _form_features(history_slice: dict[str, list[Any]], n: int = 10) -> dict[str, float]:
    """Compute rolling form features from a sliced team history."""
    recent_results = history_slice["results"][-n:]
    recent_gf = history_slice["goals_for"][-n:]
    recent_ga = history_slice["goals_against"][-n:]
    recent_points = history_slice["points"][-n:]

    if not recent_results:
        return {
            "wins": np.nan,
            "draws": np.nan,
            "losses": np.nan,
            "goals_scored": np.nan,
            "goals_conceded": np.nan,
            "goal_diff": np.nan,
            "points": np.nan,
        }

    count = len(recent_results)
    return {
        "wins": float(sum(r == "W" for r in recent_results)),
        "draws": float(sum(r == "D" for r in recent_results)),
        "losses": float(sum(r == "L" for r in recent_results)),
        "goals_scored": safe_divide(sum(recent_gf), count),
        "goals_conceded": safe_divide(sum(recent_ga), count),
        "goal_diff": safe_divide(sum(g - a for g, a in zip(recent_gf, recent_ga)), count),
        "points": safe_divide(sum(recent_points), count),
    }


def _competitive_form(history_slice: dict[str, list[Any]], n: int = 10) -> dict[str, float]:
    """Compute form from the last n competitive matches in the slice."""
    comp_indices = [
        i for i, is_comp in enumerate(history_slice["is_competitive"]) if is_comp
    ]
    if not comp_indices:
        return {"wins": np.nan, "goal_diff": np.nan}

    selected = {
        "results": [history_slice["results"][i] for i in comp_indices],
        "goals_for": [history_slice["goals_for"][i] for i in comp_indices],
        "goals_against": [history_slice["goals_against"][i] for i in comp_indices],
        "points": [history_slice["points"][i] for i in comp_indices],
        "is_competitive": [True] * len(comp_indices),
        "dates": [history_slice["dates"][i] for i in comp_indices],
    }
    recent = _form_features(selected, n=n)
    return {"wins": recent["wins"], "goal_diff": recent["goal_diff"]}


def _h2h_features(
    h2h_slice: dict[str, list[Any]], home_team: str, away_team: str
) -> dict[str, float]:
    """Compute head-to-head stats from prior meetings."""
    wins_h = wins_a = draws = 0
    goals_h: list[int] = []
    goals_a: list[int] = []

    for ht, at, hs, aws in zip(
        h2h_slice["home_team"],
        h2h_slice["away_team"],
        h2h_slice["home_score"],
        h2h_slice["away_score"],
    ):
        if ht == home_team and at == away_team:
            goals_h.append(hs)
            goals_a.append(aws)
            if hs > aws:
                wins_h += 1
            elif hs == aws:
                draws += 1
            else:
                wins_a += 1
        elif ht == away_team and at == home_team:
            goals_h.append(aws)
            goals_a.append(hs)
            if aws > hs:
                wins_h += 1
            elif aws == hs:
                draws += 1
            else:
                wins_a += 1

    total = wins_h + wins_a + draws
    return {
        "h2h_wins_H": float(wins_h),
        "h2h_wins_A": float(wins_a),
        "h2h_draws": float(draws),
        "h2h_goals_H_avg": safe_divide(sum(goals_h), len(goals_h), default=np.nan),
        "h2h_goals_A_avg": safe_divide(sum(goals_a), len(goals_a), default=np.nan),
        "h2h_total": float(total),
    }


def build_feature_matrix(df: pd.DataFrame, elo_obj: EloRating) -> pd.DataFrame:
    """Build leakage-free features for every match row."""
    enriched = add_elo_to_df(df, elo_obj)
    team_histories = _build_team_histories(enriched)
    h2h_histories = _build_h2h_histories(enriched)

    rows: list[dict[str, Any]] = []
    for row in tqdm(enriched.itertuples(index=False), total=len(enriched), desc="Features"):
        home = normalize_team_name(row.home_team)
        away = normalize_team_name(row.away_team)
        match_date = row.date

        home_slice = _slice_before(team_histories[home], match_date)
        away_slice = _slice_before(team_histories[away], match_date)
        h2h_key = frozenset({home, away})
        h2h_slice = _slice_before(h2h_histories.get(h2h_key, {"dates": []}), match_date)

        home_form = _form_features(home_slice)
        away_form = _form_features(away_slice)
        home_comp = _competitive_form(home_slice)
        away_comp = _competitive_form(away_slice)
        h2h = _h2h_features(h2h_slice, home, away)

        rows.append(
            {
                "date": match_date,
                "home_team": home,
                "away_team": away,
                "tournament": row.tournament,
                "city": row.city,
                "country": row.country,
                "result": row.result,
                "recent_wins_H": home_form["wins"],
                "recent_wins_A": away_form["wins"],
                "recent_draws_H": home_form["draws"],
                "recent_draws_A": away_form["draws"],
                "recent_losses_H": home_form["losses"],
                "recent_losses_A": away_form["losses"],
                "recent_goals_scored_H": home_form["goals_scored"],
                "recent_goals_scored_A": away_form["goals_scored"],
                "recent_goals_conceded_H": home_form["goals_conceded"],
                "recent_goals_conceded_A": away_form["goals_conceded"],
                "recent_goal_diff_H": home_form["goal_diff"],
                "recent_goal_diff_A": away_form["goal_diff"],
                "recent_points_H": home_form["points"],
                "recent_points_A": away_form["points"],
                "comp_wins_H": home_comp["wins"],
                "comp_wins_A": away_comp["wins"],
                "comp_goal_diff_H": home_comp["goal_diff"],
                "comp_goal_diff_A": away_comp["goal_diff"],
                **h2h,
                "home_elo": row.home_elo,
                "away_elo": row.away_elo,
                "elo_diff": row.elo_diff,
                "is_neutral": int(bool(row.neutral)),
                "is_world_cup": int(bool(row.is_world_cup)),
                "is_competitive": int(bool(row.is_competitive)),
                "outcome": int(row.outcome),
                "home_goals": int(row.home_score),
                "away_goals": int(row.away_score),
            }
        )

    features_df = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "features.csv"
    features_df.to_csv(out_path, index=False)

    null_counts = features_df.isna().sum()
    print(f"Feature matrix shape: {features_df.shape}")
    print("Null counts (non-zero columns):")
    print(null_counts[null_counts > 0])
    return features_df


if __name__ == "__main__":
    raw = load_results()
    cleaned = clean_data(raw)
    elo = EloRating()
    elo.compute_all_ratings(cleaned)
    build_feature_matrix(cleaned, elo)
