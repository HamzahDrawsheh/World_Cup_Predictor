"""Shared helper functions for the World Cup 2026 predictor."""

from __future__ import annotations

# Official FIFA draw (December 5, 2025) — names aligned to martj42 CSV where possible.
WC2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Map FIFA / alternate spellings to names used in martj42 results.csv.
TEAM_NAME_MAP: dict[str, str] = {
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "IR Iran": "Iran",
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Curacao": "Curaçao",
    "Cabo Verde": "Cape Verde",
}


def normalize_team_name(name: str) -> str:
    """Normalize a team name to match martj42 CSV spelling."""
    cleaned = " ".join(str(name).strip().split())
    if not cleaned:
        return cleaned
    mapped = TEAM_NAME_MAP.get(cleaned, TEAM_NAME_MAP.get(cleaned.title(), cleaned))
    return mapped


def get_all_wc2026_teams() -> list[str]:
    """Return all 48 World Cup 2026 teams using CSV-normalized names."""
    teams: list[str] = []
    for group_teams in WC2026_GROUPS.values():
        for team in group_teams:
            normalized = normalize_team_name(team)
            if normalized not in teams:
                teams.append(normalized)
    return sorted(teams)


def points_from_result(goals_for: int, goals_against: int) -> int:
    """Return league points (3/1/0) from a team's goal line."""
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Return a / b, or default when b is zero."""
    if b == 0:
        return default
    return a / b
