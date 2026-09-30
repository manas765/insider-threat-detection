import streamlit as st
import pandas as pd

# Source: Ponemon Institute / DTEX, "2026 Cost of Insider Risks: Global Report"
# (published Feb 2026, covers 2025 data).
# Average cost per incident: malicious insider $742,125; negligent insider $747,107.
# This project targets malicious insiders, so we use the malicious figure.
AVG_COST_PER_INSIDER_INCIDENT = 742_125  # USD per incident
COST_SOURCE = "Ponemon Institute / DTEX, 2026 Cost of Insider Risks: Global Report"


def render_cost_of_breach_panel():
    st.subheader("Business Impact: Cost of Breach vs. Detection Coverage")

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
        help="Average cost per malicious-insider incident x recall at this alert budget. "
             "Illustrative estimate per incident, not a guaranteed saving.",
    )

    # Note: "\\$" escapes the dollar sign so Streamlit doesn't treat text between two $ as math.
    st.caption(
        f"Cost basis: average \\${AVG_COST_PER_INSIDER_INCIDENT:,} per malicious-insider incident "
        f"(Source: {COST_SOURCE}, Feb 2026). "
        "Organizations averaged \\$19.5M per year in total insider-risk costs in 2025."
    )


if __name__ == "__main__":
    st.set_page_config(page_title="Cost of Breach — Standalone Test")
    render_cost_of_breach_panel()