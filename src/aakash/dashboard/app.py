import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../notebooks'))

import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Insider Threat Dashboard", layout="wide")
st.title("Insider Threat Detection Dashboard")
st.markdown("---")

BASE = '/Users/aakashsairam/insider-threat-detection'

@st.cache_data
def load_data():
    X_test = pd.read_csv(f'{BASE}/data/processed/X_test.csv')
    y_test = pd.read_csv(f'{BASE}/data/processed/y_test.csv').values.ravel()
    ae_scores = np.load(f'{BASE}/reports/ae_scores.npy')
    combined = pd.read_csv(f'{BASE}/reports/combined_risk_scores.csv')
    evaded = pd.read_csv(f'{BASE}/data/processed/X_test_evaded_strategy1.csv')
    return X_test, y_test, ae_scores, combined, evaded

X_test, y_test, ae_scores, combined, evaded = load_data()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Live Monitor", "Explainability", "Cost of Breach"])

with tab1:
    st.subheader("Risk Score Distribution")
    threshold = st.slider("Flag threshold", 0, 100, 70)
    combined["is_malicious"] = y_test
    flagged = combined[combined["combined_risk_score"] >= threshold]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(combined))
    col2.metric("Flagged", len(flagged))
    total_threats = int(y_test.sum())
    caught = int(flagged["is_malicious"].sum())
    col3.metric("Threats Caught", f"{caught}/{total_threats} ({round(caught/total_threats*100,1)}%)")
    st.dataframe(flagged[["combined_risk_score","iso_forest_scaled","oc_svm_scaled","autoencoder_scaled","is_malicious"]].sort_values("combined_risk_score", ascending=False).head(50))
    st.markdown("---")
    st.subheader("Alert Fatigue Simulation")
    thresholds = range(0, 101, 5)
    workload, recall_list = [], []
    for t in thresholds:
        flagged_t = combined[combined["combined_risk_score"] >= t]
        caught_t = int(flagged_t["is_malicious"].sum())
        workload.append(len(flagged_t))
        recall_list.append(round(caught_t / total_threats * 100, 1) if total_threats > 0 else 0)
    fatigue_df = pd.DataFrame({"threshold": list(thresholds), "alerts_to_review": workload, "threats_caught_pct": recall_list})
    col_c, col_d = st.columns(2)
    with col_c:
        st.write("Alerts to review")
        st.line_chart(fatigue_df.set_index("threshold")["alerts_to_review"])
    with col_d:
        st.write("% of threats caught")
        st.line_chart(fatigue_df.set_index("threshold")["threats_caught_pct"])

with tab2:
    st.subheader("Live Monitoring Simulation")
    mode = st.radio("Dataset", ["Normal", "Evaded (Strategy 1 — evasion attack)"], horizontal=True)
    speed = st.slider("Speed (rows/sec)", 1, 20, 5)
    if st.button("Start Live Feed"):
        data = combined.head(1000).copy() if mode == "Normal" else evaded.head(1000).copy()
        if "combined_risk_score" not in data.columns:
            st.error("Evaded dataset missing combined_risk_score column.")
            st.stop()
        alert_thresh = 70
        metrics_box = st.empty()
        chart_box = st.empty()
        feed_box = st.empty()
        seen, alerts, threats_caught = [], 0, 0
        for i, row in data.iterrows():
            score = float(row["combined_risk_score"])
            label = int(row.get("true_label", row.get("is_malicious", 0)))
            seen.append({"index": len(seen), "risk_score": score})
            if score >= alert_thresh:
                alerts += 1
                if label == 1:
                    threats_caught += 1
            if len(seen) % 10 == 0 or i == len(data) - 1:
                df_seen = pd.DataFrame(seen)
                with metrics_box.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Processed", len(seen))
                    m2.metric("Alerts", alerts)
                    m3.metric("Threats Caught", threats_caught)
                chart_box.line_chart(df_seen.set_index("index")["risk_score"])
                if score >= alert_thresh:
                    feed_box.warning(f"Row {len(seen)}: risk={score:.1f} | malicious={label}")
            time.sleep(1 / speed)
        st.success(f"Done — {threats_caught} threats caught.")

with tab3:
    st.subheader("SHAP Feature Importance")
    import traceback
    try:
        import shap, joblib
        model_path = f'{BASE}/models/isolation_forest.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            sample = X_test.head(200)
            explainer = shap.Explainer(model, sample)
            shap_values = explainer(sample)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            shap.plots.bar(shap_values, max_display=10, show=False)
            st.pyplot(fig)
        else:
            st.info("Model file not found.")
    except Exception:
        st.error(traceback.format_exc())

with tab4:
    import traceback
    try:
        from cost_of_breach_panel import render_cost_of_breach_panel
        render_cost_of_breach_panel()
    except Exception:
        st.error(traceback.format_exc())
