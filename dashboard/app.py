"""Streamlit dashboard for the World Cup 2026 predictor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import clean_data, load_results
from src.elo import EloRating
from src.model import build_feature_lookup, load_models, predict_match
from src.simulator import TournamentSimulator
from src.utils import get_all_wc2026_teams, normalize_team_name

MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
FOOTER = (
    "Data: github.com/martj42/international_results (CC0) | "
    "Built with Cursor AI + Streamlit"
)

CONFEDERATIONS = {
    "UEFA": {
        "Czech Republic", "Bosnia and Herzegovina", "Switzerland", "Scotland", "Turkey",
        "Germany", "Netherlands", "Sweden", "Belgium", "Spain", "France", "Norway",
        "Austria", "Portugal", "Croatia", "England",
    },
    "CONMEBOL": {"Brazil", "Paraguay", "Ecuador", "Uruguay", "Argentina", "Colombia", "Chile", "Peru"},
    "CONCACAF": {"Mexico", "Canada", "United States", "Haiti", "Panama", "Curacao"},
    "CAF": {
        "South Africa", "Morocco", "Ivory Coast", "Senegal", "Egypt", "Cabo Verde",
        "Algeria", "DR Congo", "Ghana", "Tunisia",
    },
    "AFC": {"South Korea", "Qatar", "Japan", "Iran", "Saudi Arabia", "Iraq", "Jordan", "Uzbekistan"},
    "OFC": {"New Zealand", "Australia"},
}


def confederation(team: str) -> str:
    team = normalize_team_name(team)
    for conf, teams in CONFEDERATIONS.items():
        if team in teams:
            return conf
    return "Other"


@st.cache_data(show_spinner=False)
def load_match_data() -> pd.DataFrame:
    return clean_data(load_results())


@st.cache_data(show_spinner=False)
def load_features_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.csv"
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_resource(show_spinner=False)
def load_prediction_assets():
    models = load_models()
    features_df = load_features_data()
    lookup = build_feature_lookup(features_df)
    return models, lookup


@st.cache_data(show_spinner=False)
def compute_elo_history() -> pd.DataFrame:
    df = load_match_data()
    elo = EloRating()
    elo.compute_all_ratings(df)
    history = elo.get_ratings_history_df()
    long_rows = []
    for _, row in history.iterrows():
        long_rows.append({"date": row["date"], "team": row["home_team"], "elo": row["home_elo_after"]})
        long_rows.append({"date": row["date"], "team": row["away_team"], "elo": row["away_elo_after"]})
    return pd.DataFrame(long_rows)


def _render_match_outcome_metrics(
    team1: str,
    team2: str,
    prob_team1: float,
    prob_draw: float,
    prob_team2: float,
) -> tuple[str, float]:
    """Render outcome metrics and return the label + probability of the most likely outcome."""
    outcomes = [
        (f"{team1} wins", prob_team1),
        ("Draw", prob_draw),
        (f"{team2} wins", prob_team2),
    ]
    best_label, best_prob = max(outcomes, key=lambda x: x[1])

    cols = st.columns(3)
    for col, (label, prob) in zip(cols, outcomes):
        highlight = label == best_label
        with col:
            if highlight:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #0f5132 0%, #198754 100%);
                        border: 2px solid #75b798;
                        border-radius: 12px;
                        padding: 1rem 1.25rem;
                        text-align: center;
                        box-shadow: 0 4px 14px rgba(25, 135, 84, 0.35);
                    ">
                        <div style="color: #d1e7dd; font-size: 0.85rem; margin-bottom: 0.25rem;">
                            ★ Most likely
                        </div>
                        <div style="color: #ffffff; font-size: 2rem; font-weight: 700;">
                            {100 * prob:.1f}%
                        </div>
                        <div style="color: #ffffff; font-size: 1rem; font-weight: 600;">
                            {label}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.metric(label, f"{100 * prob:.1f}%")

    return best_label, best_prob


def page_match_predictor() -> None:
    st.header("Match Predictor")
    st.markdown(
        "Compare any two World Cup 2026 teams. "
        "**Team 1** and **Team 2** are just labels — with neutral venue on, "
        "there is no home advantage (standard for World Cup matches)."
    )

    teams = get_all_wc2026_teams()
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    with col2:
        team2 = st.selectbox("Team 2", teams, index=teams.index("Morocco") if "Morocco" in teams else 1)

    if team1 == team2:
        st.warning("Team 1 and Team 2 must be different.")

    neutral = st.checkbox(
        "Neutral venue (recommended for World Cup)",
        value=True,
        help="Neutral venue removes home-field advantage from the model.",
    )

    models, lookup = load_prediction_assets()

    if st.button("Predict", type="primary", disabled=(team1 == team2)):
        xgb_model, poisson_home, poisson_away, scaler, imputer, poisson_imputer, poisson_scaler, feature_columns = models
        result = predict_match(
            team1, team2, lookup, xgb_model, poisson_home, poisson_away,
            scaler, imputer, poisson_imputer, poisson_scaler, feature_columns,
            neutral=neutral, is_world_cup=True,
        )

        best_label, best_prob = _render_match_outcome_metrics(
            team1,
            team2,
            result["prob_home_win"],
            result["prob_draw"],
            result["prob_away_win"],
        )

        if best_label == "Draw":
            headline = f"{100 * best_prob:.1f}% — Draw"
            detail = f"A draw is the most likely result between **{team1}** and **{team2}**."
        elif best_label == f"{team1} wins":
            headline = f"{100 * best_prob:.1f}% — {team1} wins"
            detail = f"**{team1}** is favored over **{team2}**."
        else:
            headline = f"{100 * best_prob:.1f}% — {team2} wins"
            detail = f"**{team2}** is favored over **{team1}**."

        st.markdown(f"### Prediction: {headline}")
        st.info(detail)

        outcome_labels = [f"{team1} wins", "Draw", f"{team2} wins"]
        probabilities = [
            result["prob_home_win"],
            result["prob_draw"],
            result["prob_away_win"],
        ]
        colors = [
            "#198754" if label == best_label else "#adb5bd"
            for label in outcome_labels
        ]
        prob_df = pd.DataFrame({"Outcome": outcome_labels, "Probability": probabilities})
        fig = px.bar(
            prob_df,
            x="Probability",
            y="Outcome",
            orientation="h",
            range_x=[0, 1],
            text=prob_df["Probability"].map(lambda p: f"{100 * p:.1f}%"),
        )
        fig.update_traces(marker_color=colors, textposition="outside")
        fig.update_layout(
            title="Win / Draw / Loss probabilities",
            xaxis_tickformat=".0%",
            showlegend=False,
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

        score_col1, score_col2 = st.columns(2)
        with score_col1:
            st.metric("Expected score", f"{result['expected_home_goals']:.1f} – {result['expected_away_goals']:.1f}")
            st.caption(f"{team1} – {team2}")
        with score_col2:
            st.metric("Most likely scoreline", result["most_likely_score"])
            st.caption("Highest-probability exact score (Poisson model)")

        score_parts = result["most_likely_score"].split("-")
        poisson_suggests_draw = (
            len(score_parts) == 2 and score_parts[0] == score_parts[1]
        )
        goals_very_close = (
            abs(result["expected_home_goals"] - result["expected_away_goals"]) < 0.25
        )
        if (poisson_suggests_draw or goals_very_close) and best_label != "Draw":
            st.caption(
                "ℹ️ **Why not a draw?** The win/draw/loss bar uses a separate model "
                "(form, Elo, history). The score uses expected goals only. "
                f"Here all three outcomes are close ({100 * result['prob_home_win']:.0f}% / "
                f"{100 * result['prob_draw']:.0f}% / {100 * result['prob_away_win']:.0f}%) — "
                f"a tight game where **{best_label.replace(' wins', '')}** has a small edge, "
                "not a clear favorite."
            )

        with st.expander("Head-to-head history (last 10 meetings)"):
            df = load_match_data()
            h2h = df[
                ((df["home_team"] == normalize_team_name(team1)) & (df["away_team"] == normalize_team_name(team2)))
                | ((df["home_team"] == normalize_team_name(team2)) & (df["away_team"] == normalize_team_name(team1)))
            ].sort_values("date", ascending=False).head(10)
            if h2h.empty:
                st.write("No previous meetings found in the dataset.")
            else:
                display = h2h[["date", "home_team", "away_team", "home_score", "away_score", "tournament"]].copy()
                display["Score"] = display["home_score"].astype(str) + "-" + display["away_score"].astype(str)
                display = display.rename(
                    columns={
                        "date": "Date",
                        "home_team": "Home (historical)",
                        "away_team": "Away (historical)",
                        "tournament": "Tournament",
                    }
                )[["Date", "Home (historical)", "Away (historical)", "Score", "Tournament"]]
                st.dataframe(display, use_container_width=True, hide_index=True)


def page_tournament_odds() -> None:
    st.header("Tournament Odds")
    n_sims = st.slider("Simulations", min_value=1000, max_value=20000, value=5000, step=1000)

    if st.button("Run Simulation"):
        with st.spinner(f"Running {n_sims:,} simulations..."):
            models, lookup = load_prediction_assets()
            sim = TournamentSimulator(models, lookup)
            results = sim.run_simulation(n_simulations=n_sims)
            results["confederation"] = results["team"].map(confederation)

            treemap = px.treemap(
                results,
                path=["confederation", "team"],
                values="champion_pct",
                color="confederation",
                title="Championship Probability",
            )
            st.plotly_chart(treemap, use_container_width=True)
            st.dataframe(
                results[
                    ["team", "champion_pct", "final_pct", "semi_pct", "quarter_pct", "group_exit_pct"]
                ].style.format(
                    {
                        "champion_pct": "{:.2f}%",
                        "final_pct": "{:.2f}%",
                        "semi_pct": "{:.2f}%",
                        "quarter_pct": "{:.2f}%",
                        "group_exit_pct": "{:.2f}%",
                    }
                )
            )


def page_team_profile() -> None:
    st.header("Team Profile")
    teams = get_all_wc2026_teams()
    team = st.selectbox("Team", teams)

    elo_hist = compute_elo_history()
    team_elo = elo_hist[elo_hist["team"] == normalize_team_name(team)].sort_values("date")
    current_elo = team_elo["elo"].iloc[-1] if not team_elo.empty else 1500.0

    c1, c2 = st.columns([1, 2])
    c1.metric("Current Elo", f"{current_elo:.0f}")

    cutoff = pd.Timestamp("2021-01-01")
    recent_elo = team_elo[team_elo["date"] >= cutoff]
    if not recent_elo.empty:
        fig = px.line(recent_elo, x="date", y="elo", title=f"{team} Elo trend (last 5 years)")
        c2.plotly_chart(fig, use_container_width=True)

    df = load_match_data()
    matches = df[
        (df["home_team"] == normalize_team_name(team)) | (df["away_team"] == normalize_team_name(team))
    ].sort_values("date", ascending=False)

    last10 = matches.head(10).copy()
    rows = []
    for _, m in last10.iterrows():
        if m["home_team"] == normalize_team_name(team):
            opp, gf, ga = m["away_team"], m["home_score"], m["away_score"]
        else:
            opp, gf, ga = m["home_team"], m["away_score"], m["home_score"]
        res = "D" if gf == ga else ("W" if gf > ga else "L")
        rows.append({"date": m["date"], "opponent": opp, "score": f"{gf}-{ga}", "result": res, "tournament": m["tournament"]})
    st.subheader("Last 10 results")
    st.dataframe(pd.DataFrame(rows))

    last20 = matches.head(20).copy()
    gs, gc, dates = [], [], []
    for _, m in last20.iterrows():
        if m["home_team"] == normalize_team_name(team):
            gs.append(m["home_score"])
            gc.append(m["away_score"])
        else:
            gs.append(m["away_score"])
            gc.append(m["home_score"])
        dates.append(m["date"])
    trend = pd.DataFrame({"date": dates, "goals_scored": gs, "goals_conceded": gc})
    fig2 = px.line(trend, x="date", y=["goals_scored", "goals_conceded"], title="Goals scored vs conceded (last 20)")
    st.plotly_chart(fig2, use_container_width=True)


def page_model_report() -> None:
    st.header("Model Report")
    metrics_path = MODELS_DIR / "eval_metrics.json"
    if not metrics_path.exists():
        st.warning("No evaluation metrics found. Run src/model.py first.")
        return

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{100 * metrics['accuracy']:.2f}%")
    c2.metric("Log Loss", f"{metrics['log_loss']:.4f}")
    c3.metric("Brier Score (mean)", f"{metrics['brier_score']['mean']:.4f}")

    cm = metrics["confusion_matrix"]
    labels = ["Home Win", "Draw", "Away Win"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

    imp = pd.DataFrame(
        list(metrics["feature_importances"].items()),
        columns=["feature", "importance"],
    ).sort_values("importance", ascending=True).tail(20)
    fig2 = px.bar(imp, x="importance", y="feature", orientation="h", title="Top 20 Feature Importances")
    st.plotly_chart(fig2, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="WC 2026 Predictor", layout="wide")
    st.title("World Cup 2026 Predictor")

    page = st.sidebar.radio(
        "Navigation",
        ["Match Predictor", "Tournament Odds", "Team Profile", "Model Report"],
    )

    if page == "Match Predictor":
        page_match_predictor()
    elif page == "Tournament Odds":
        page_tournament_odds()
    elif page == "Team Profile":
        page_team_profile()
    else:
        page_model_report()

    st.markdown("---")
    st.caption(FOOTER)


if __name__ == "__main__":
    main()
