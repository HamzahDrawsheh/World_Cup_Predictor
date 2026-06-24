"""Monte Carlo World Cup 2026 tournament simulator."""

from __future__ import annotations

import random
import sys
import time
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import build_feature_lookup, load_models, predict_match
from src.utils import WC2026_GROUPS, normalize_team_name, points_from_result

PROCESSED_DIR = ROOT / "data" / "processed"

ROUND_LABELS = ["group_exit", "r32", "r16", "quarter", "semi", "final", "champion"]


class TournamentSimulator:
    """Simulate World Cup 2026 group and knockout stages."""

    def __init__(self, models_tuple, feature_lookup: dict) -> None:
        (
            self.xgb_model,
            self.poisson_home,
            self.poisson_away,
            self.scaler,
            self.imputer,
            self.poisson_imputer,
            self.poisson_scaler,
            self.feature_columns,
        ) = models_tuple
        self.feature_lookup = feature_lookup

    def _predict(self, home_team: str, away_team: str, neutral: bool = True) -> dict:
        return predict_match(
            home_team=home_team,
            away_team=away_team,
            feature_lookup=self.feature_lookup,
            xgb_model=self.xgb_model,
            poisson_home=self.poisson_home,
            poisson_away=self.poisson_away,
            scaler=self.scaler,
            imputer=self.imputer,
            poisson_imputer=self.poisson_imputer,
            poisson_scaler=self.poisson_scaler,
            feature_columns=self.feature_columns,
            neutral=neutral,
            is_world_cup=True,
        )

    def simulate_match(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        knockout: bool = False,
    ) -> dict:
        """Simulate a single match using model probabilities."""
        home_team = normalize_team_name(home_team)
        away_team = normalize_team_name(away_team)
        pred = self._predict(home_team, away_team, neutral=neutral)

        p_home = pred["prob_home_win"]
        p_draw = pred["prob_draw"]
        p_away = pred["prob_away_win"]

        if knockout:
            p_draw *= 0.6
            remainder = p_home + p_away
            if remainder > 0:
                scale = (1.0 - p_draw) / remainder
                p_home *= scale
                p_away *= scale
            else:
                p_home, p_away = 0.5, 0.5

        outcomes = ["home", "draw", "away"]
        probs = [p_home, p_draw, p_away]
        chosen = random.choices(outcomes, weights=probs, k=1)[0]

        if chosen == "home":
            home_goals = max(0, int(np.random.poisson(pred["expected_home_goals"])))
            away_goals = min(home_goals - 1, int(np.random.poisson(pred["expected_away_goals"]))) if home_goals > 0 else 0
            if home_goals <= away_goals:
                home_goals = away_goals + 1
            winner = home_team
            is_draw = False
        elif chosen == "away":
            away_goals = max(0, int(np.random.poisson(pred["expected_away_goals"])))
            home_goals = min(away_goals - 1, int(np.random.poisson(pred["expected_home_goals"]))) if away_goals > 0 else 0
            if away_goals <= home_goals:
                away_goals = home_goals + 1
            winner = away_team
            is_draw = False
        else:
            home_goals = int(np.random.poisson(pred["expected_home_goals"]))
            away_goals = home_goals
            if knockout:
                if random.random() < 0.5:
                    winner = home_team
                else:
                    winner = away_team
                is_draw = False
            else:
                winner = None
                is_draw = True

        return {
            "winner": winner,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "draw": is_draw,
        }

    def simulate_group(self, teams: list[str]) -> pd.DataFrame:
        """Simulate round-robin group stage and return sorted standings."""
        teams = [normalize_team_name(t) for t in teams]
        records = {t: {"team": t, "played": 0, "wins": 0, "draws": 0, "losses": 0,
                       "goals_for": 0, "goals_against": 0, "goal_diff": 0, "points": 0}
                 for t in teams}

        for home, away in combinations(teams, 2):
            result = self.simulate_match(home, away, neutral=True, knockout=False)
            hg, ag = result["home_goals"], result["away_goals"]
            for team, gf, ga in [(home, hg, ag), (away, ag, hg)]:
                records[team]["played"] += 1
                records[team]["goals_for"] += gf
                records[team]["goals_against"] += ga
                records[team]["goal_diff"] = records[team]["goals_for"] - records[team]["goals_against"]
                records[team]["points"] += points_from_result(gf, ga)
            if hg > ag:
                records[home]["wins"] += 1
                records[away]["losses"] += 1
            elif hg < ag:
                records[away]["wins"] += 1
                records[home]["losses"] += 1
            else:
                records[home]["draws"] += 1
                records[away]["draws"] += 1

        standings = pd.DataFrame(records.values())
        standings = standings.sort_values(
            ["points", "goal_diff", "goals_for"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        return standings

    def get_third_place_qualifiers(self, all_standings: dict[str, pd.DataFrame]) -> list[str]:
        """Return the best eight third-place teams across all groups."""
        third_places: list[dict] = []
        for standings in all_standings.values():
            if len(standings) >= 3:
                row = standings.iloc[2]
                third_places.append(row.to_dict())
        third_df = pd.DataFrame(third_places)
        third_df = third_df.sort_values(
            ["points", "goal_diff", "goals_for"],
            ascending=[False, False, False],
        )
        return third_df.head(8)["team"].tolist()

    def simulate_knockout_round(self, teams: list[str]) -> list[str]:
        """Simulate one knockout round from an ordered list of paired teams."""
        winners: list[str] = []
        for i in range(0, len(teams), 2):
            home, away = teams[i], teams[i + 1]
            result = self.simulate_match(home, away, neutral=True, knockout=True)
            winners.append(result["winner"])
        return winners

    def simulate_tournament(self, groups: dict[str, list[str]] | None = None) -> tuple[str, dict[str, int]]:
        """Run one full tournament and return winner plus max round per team."""
        groups = deepcopy(groups or WC2026_GROUPS)
        all_teams = [normalize_team_name(t) for grp in groups.values() for t in grp]
        max_round = {team: 0 for team in all_teams}

        all_standings: dict[str, pd.DataFrame] = {}
        qualifiers: list[str] = []

        for letter, teams in groups.items():
            standings = self.simulate_group(teams)
            all_standings[letter] = standings
            first, second = standings.iloc[0]["team"], standings.iloc[1]["team"]
            qualifiers.extend([first, second])
            max_round[first] = max(max_round[first], 1)
            max_round[second] = max(max_round[second], 1)

        third_qualifiers = self.get_third_place_qualifiers(all_standings)
        for team in third_qualifiers:
            max_round[team] = max(max_round[team], 1)
        qualifiers.extend(third_qualifiers)

        bracket = qualifiers[:32]
        round_sizes = [32, 16, 8, 4, 2]
        round_levels = [1, 2, 3, 4, 5]

        for size, level in zip(round_sizes, round_levels):
            if len(bracket) < size:
                break
            current = bracket[:size]
            winners = self.simulate_knockout_round(current)
            for team in winners:
                max_round[team] = max(max_round[team], level)
            if size == 2:
                champion = winners[0]
                max_round[champion] = 6
                runner_up = [t for t in current if t != champion][0]
                max_round[runner_up] = max(max_round[runner_up], 5)
                return champion, max_round
            bracket = winners

        return bracket[0], max_round

    def run_simulation(
        self, n_simulations: int = 10000, groups: dict[str, list[str]] | None = None
    ) -> pd.DataFrame:
        """Run many tournament simulations and aggregate exit-round rates."""
        groups = groups or WC2026_GROUPS
        all_teams = sorted({normalize_team_name(t) for grp in groups.values() for t in grp})
        counts = {team: {label: 0 for label in ROUND_LABELS} for team in all_teams}

        for _ in range(n_simulations):
            _, max_round = self.simulate_tournament(groups)
            for team in all_teams:
                level = max_round.get(team, 0)
                if level == 0:
                    counts[team]["group_exit"] += 1
                if level >= 1:
                    counts[team]["r32"] += 1
                if level >= 2:
                    counts[team]["r16"] += 1
                if level >= 3:
                    counts[team]["quarter"] += 1
                if level >= 4:
                    counts[team]["semi"] += 1
                if level >= 5:
                    counts[team]["final"] += 1
                if level >= 6:
                    counts[team]["champion"] += 1

        rows = []
        for team, stats in counts.items():
            rows.append(
                {
                    "team": team,
                    "champion_pct": 100.0 * stats["champion"] / n_simulations,
                    "final_pct": 100.0 * stats["final"] / n_simulations,
                    "semi_pct": 100.0 * stats["semi"] / n_simulations,
                    "quarter_pct": 100.0 * stats["quarter"] / n_simulations,
                    "r16_pct": 100.0 * stats["r16"] / n_simulations,
                    "r32_pct": 100.0 * stats["r32"] / n_simulations,
                    "group_exit_pct": 100.0 * stats["group_exit"] / n_simulations,
                }
            )

        return pd.DataFrame(rows).sort_values("champion_pct", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    start = time.time()
    models = load_models()
    features_df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    lookup = build_feature_lookup(features_df)
    sim = TournamentSimulator(models, lookup)

    results = sim.run_simulation(n_simulations=10000)
    elapsed = time.time() - start
    print("Top 20 teams by championship probability:")
    print(results.head(20).to_string(index=False))
    print(f"\nEstimated runtime for 10,000 simulations: {elapsed:.1f}s")
