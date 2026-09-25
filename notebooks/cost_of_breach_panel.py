import streamlit as st
import pandas as pd

def render_cost_of_breach_panel():
    st.subheader("Business Impact: Cost of Breach vs. Detection Coverage")

    # TODO: Akshata -- replace with the real cited statistic once sourced
    AVG_COST_PER_INSIDER_INCIDENT = 750_000  # placeholder, USD

    alert_fatigue_iso = pd.read_csv("reports/alert_fatigue_isoforest.csv")
    alert_fatigue_svm = pd.read_csv("reports/alert_fatigue_ocsvm.csv")

    st.caption("Move the slider to see the tradeoff between analyst workload and threat coverage.")

    max_alerts = int(max(alert_fatigue_iso["alerts"].max(), alert_fatigue_svm["alerts"].max()))
    alerts_budget = st.slider("Alerts an analyst reviews per day", 0, max_alerts, value=5000, step=100)

    def recall_at(df, budget):
        row = df[df["alerts"] <= budget].tail(1)
        return float(row["recall"].values[0]) if not row.empty else 0.0

    recall_iso = recall_at(alert_fatigue_iso, alerts_budget)
    recall_svm = recall_at(alert_fatigue_svm, alerts_budget)
    best_recall = max(recall_iso, recall_svm)

    col1, col2, col3 = st.columns(3)
    col1.metric("Threats caught", f"{best_recall * 100:.1f}%")
    col2.metric("Threats missed", f"{(1 - best_recall) * 100:.1f}%")
    col3.metric(
        "Est. exposure reduction",
        f"${AVG_COST_PER_INSIDER_INCIDENT * best_recall:,.0f}",
        help="Average cost per incident x recall at this alert budget",
    )

    st.caption(
        f"Based on an estimated average cost of ${AVG_COST_PER_INSIDER_INCIDENT:,} per insider incident. "
        "Replace this with Akshata's cited source before the pitch."
    )