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

# ── TAB 1: OVERVIEW ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Risk Score Distribution")
    threshold = st.slider("Flag threshold", 0, 100, 70)
    combined['is_malicious'] = y_test
    flagged = combined[combined['combined_risk_score'] >= threshold]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(combined))
    col2.metric("Flagged", len(flagged))
    total_threats = int(y_test.sum())
    caught = int(flagged['is_malicious'].sum())
    col3.metric("Threats Caught", f"{caught}/{total_threats} ({round(caught/total_threats*100,1)}%)")
    st.dataframe(flagged[['combined_risk_score','iso_forest_scaled','oc_svm_scaled','autoencoder_scaled','is_malicious']].sort_values('combined_risk_score', ascending=False).head(50))

    st.markdown("---")
    st.subheader("Alert Fatigue Simulation")
    thresholds = range(0, 101, 5)
    workload, recall_list = [], []
    for t in thresholds:
        flagged_t = combined[combined['combined_risk_score'] >= t]
        caught_t = int(flagged_t['is_malicious'].sum())
        workload.append(len(flagged_t))
        recall_list.append(round(caught_t / total_threats * 100, 1) if total_threats > 0 else 0)
    fatigue_df = pd.DataFrame({'threshold': list(thresholds), 'alerts_to_review': workload, 'threats_caught_pct': recall_list})
    col_c, col_d = st.columns(2)
    with col_c:
        st.write("Alerts to review")
        st.line_chart(fatigue_df.set_index('threshold')['alerts_to_review'])
    with col_d:
        st.write("% of threats caught")
        st.line_chart(fatigue_df.set_index('threshold')['threats_caught_pct'])


    st.markdown("---")
    st.subheader("Role-Based Behavioral Analysis")
    st.caption("Select a flagged employee to see role context")

    # TODO: Replace mock_users with real user IDs from Pushkar's role-feature CSV
    mock_users = [f"User_{i:04d}" for i in range(1, 11)]
    selected_user = st.selectbox("Select Employee", mock_users)

    # TODO: Replace this block with real role columns from Pushkar's output
    # Expected columns: employee_role, role_usb_zscore, role_files_zscore,
    #                   role_session_zscore, role_email_zscore
    import random
    random.seed(hash(selected_user) % 1000)
    mock_role_data = {
        "role": random.choice(["Finance", "IT", "HR", "Engineering", "Sales"]),
        "risk_score": random.randint(55, 95),
        "usb_deviation": random.choice(["HIGH", "HIGH", "NORMAL", "LOW"]),
        "file_deviation": random.choice(["HIGH", "HIGH", "NORMAL", "LOW"]),
        "session_deviation": random.choice(["NORMAL", "LOW", "HIGH"]),
        "email_deviation": random.choice(["HIGH", "NORMAL", "NORMAL"]),
    }

    def deviation_badge(level):
        colors = {"HIGH": "🔴", "NORMAL": "🟡", "LOW": "🟢"}
        return colors.get(level, "⚪")

    st.markdown(f"""
**Employee:** `{selected_user}`  
**Current Role:** `{mock_role_data["role"]}`  
**Combined Risk Score:** `{mock_role_data["risk_score"]}/100`
""")

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**Role Behaviour Deviation**")
        st.markdown(f"{deviation_badge(mock_role_data['usb_deviation'])} USB Activity — **{mock_role_data['usb_deviation']}**")
        st.markdown(f"{deviation_badge(mock_role_data['file_deviation'])} File Access — **{mock_role_data['file_deviation']}**")
        st.markdown(f"{deviation_badge(mock_role_data['session_deviation'])} Session Duration — **{mock_role_data['session_deviation']}**")
        st.markdown(f"{deviation_badge(mock_role_data['email_deviation'])} External Email — **{mock_role_data['email_deviation']}**")

    with rc2:
        st.markdown("**Why Flagged?**")
        high_features = [k.replace("_deviation","").replace("_"," ").title()
                         for k, v in mock_role_data.items() if v == "HIGH" and "deviation" in k]
        if high_features:
            for feat in high_features:
                st.warning(f"{feat} is significantly above the normal range for the **{mock_role_data['role']}** role.")
        else:
            st.success("Behavior is within expected range for this role.")

# ── TAB 2: LIVE MONITOR ───────────────────────────────────────────────────────
with tab2:
    st.subheader("Live Monitoring Simulation")
    mode = st.radio("Dataset", ["Normal", "Evaded (Strategy 1 — evasion attack)"], horizontal=True)
    speed = st.slider("Speed (rows/sec)", 1, 20, 5)

    if st.button("▶ Start Live Feed"):
        data = combined.head(1000).copy() if mode == "Normal" else evaded.head(1000).copy()
        if 'combined_risk_score' not in data.columns:
            st.error("Evaded dataset missing combined_risk_score column — check Manas's file.")
            st.stop()

        alert_thresh = 70
        metrics_box = st.empty()
        chart_box = st.empty()
        feed_box = st.empty()

        seen, alerts, threats_caught = [], 0, 0
        for i, row in data.iterrows():
            score = float(row['combined_risk_score'])
            label = int(row.get('true_label', row.get('is_malicious', 0)))
            seen.append({'index': len(seen), 'risk_score': score})
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
                chart_box.line_chart(df_seen.set_index('index')['risk_score'])
                if score >= alert_thresh:
                    feed_box.warning(f"🚨 Row {len(seen)}: risk={score:.1f} | malicious={label}")

            time.sleep(1 / speed)

        st.success(f"Done — {threats_caught} threats caught.")

# ── TAB 3: EXPLAINABILITY ─────────────────────────────────────────────────────
with tab3:
    st.subheader("SHAP Feature Importance")
    try:
        import shap, joblib
        import traceback
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
            st.info("Model file not found at models/isolation_forest.pkl — run train_and_save_models.py first.")
    except Exception as e:
        import traceback
        st.error(traceback.format_exc())

# ── TAB 4: COST OF BREACH ─────────────────────────────────────────────────────
with tab4:
    try:
        from cost_of_breach_panel import render_cost_of_breach_panel
        render_cost_of_breach_panel()
    except Exception as e:
        import traceback
        st.error(traceback.format_exc())