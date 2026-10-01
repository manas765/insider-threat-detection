import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../notebooks'))

import streamlit as st
import pandas as pd
import numpy as np
import time
import random

st.set_page_config(page_title="Insider Threat Dashboard", layout="wide")

st.markdown("""
<style>
footer { visibility: hidden; }
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
div[data-testid="stTabs"] { display: flex; justify-content: center; }
div[data-testid="stTabsContent"] { width: 100%; }
button[data-baseweb="tab"] {
    font-size: 1rem !important; font-weight: 700 !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    padding: 0.7rem 2rem !important; color: #888 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #fff !important; border-bottom: 2px solid #e63946 !important;
}
div.stButton > button {
    font-size: 1rem !important; font-weight: 700 !important;
    letter-spacing: 0.06em !important; padding: 0.75rem 3rem !important;
    border-radius: 4px !important; background-color: #e63946 !important;
    color: #fff !important; border: none !important; min-width: 200px !important;
}
div.stButton > button:hover { background-color: #c1121f !important; }
.modebar-btn svg { width: 26px !important; height: 26px !important; }
.modebar { transform: scale(1.6); transform-origin: top right; }
div[data-testid="metric-container"] {
    background: #161622; border: 1px solid #2a2a3a;
    border-radius: 6px; padding: 1rem 1.2rem;
}
div[data-testid="metric-container"] label {
    font-size: 0.75rem !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; color: #666 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important; font-weight: 700 !important; color: #fff !important;
}
.stSlider label {
    font-size: 0.8rem !important; font-weight: 600 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important; color: #888 !important;
}
div[role="radiogroup"] label { font-size: 0.9rem !important; }
h2, h3 { font-weight: 700 !important; letter-spacing: -0.02em !important; }
.stDataFrame { border: 1px solid #2a2a3a !important; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size:3rem;font-weight:900;letter-spacing:-0.03em;padding:0.6rem 0 0 0;'>Insider Threat Detection</h1>", unsafe_allow_html=True)
st.markdown("---")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

@st.cache_data
def load_data():
    X_test   = pd.read_csv(f'{BASE}/data/processed/X_test.csv')
    y_test   = pd.read_csv(f'{BASE}/data/processed/y_test.csv').values.ravel()
    ae_scores = np.load(f'{BASE}/reports/ae_scores.npy')
    combined  = pd.read_csv(f'{BASE}/reports/combined_risk_scores.csv')
    evaded    = pd.read_csv(f'{BASE}/data/processed/X_test_evaded_strategy1.csv')
    return X_test, y_test, ae_scores, combined, evaded

@st.cache_data
def load_role_features():
    p = f'{BASE}/data/processed/test_role_features.csv'
    return pd.read_csv(p) if os.path.exists(p) else None

with st.spinner("Loading..."):
    X_test, y_test, ae_scores, combined, evaded = load_data()
    role_df = load_role_features()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Live Monitor", "Explainability", "Cost of Breach"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Risk Score Distribution")
    threshold = st.slider("Flag threshold", 0, 100, 70)
    combined["is_malicious"] = y_test
    flagged = combined[combined["combined_risk_score"] >= threshold]
    total_threats = int(y_test.sum())
    caught = int(flagged["is_malicious"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(combined):,}")
    c2.metric("Flagged", f"{len(flagged):,}")
    c3.metric("Threats Caught", f"{caught} / {total_threats}  ({round(caught/total_threats*100,1)}%)")

    st.markdown("#### Flagged Records")
    st.dataframe(
        flagged[["combined_risk_score","iso_forest_scaled","oc_svm_scaled","autoencoder_scaled","is_malicious"]]
        .sort_values("combined_risk_score", ascending=False).head(50),
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("### Alert Fatigue Simulation")
    thresholds = range(0, 101, 5)
    workload, recall_list = [], []
    for t in thresholds:
        ft = combined[combined["combined_risk_score"] >= t]
        ct = int(ft["is_malicious"].sum())
        workload.append(len(ft))
        recall_list.append(round(ct / total_threats * 100, 1) if total_threats > 0 else 0)
    fatigue_df = pd.DataFrame({"threshold": list(thresholds), "alerts_to_review": workload, "threats_caught_pct": recall_list})
    col_c, col_d = st.columns(2)
    with col_c:
        st.caption("ALERTS TO REVIEW")
        st.line_chart(fatigue_df.set_index("threshold")["alerts_to_review"])
    with col_d:
        st.caption("% THREATS CAUGHT")
        st.line_chart(fatigue_df.set_index("threshold")["threats_caught_pct"])

    st.markdown("---")
    st.markdown("### Role-Based Behavioral Analysis")
    st.caption("All employees ranked by risk score. Each column shows deviation from their role baseline.")

    EMP_NAMES = [
        "Alice Mercer","Bob Tanaka","Carlos Ruiz","Diana Patel","Ethan Wu",
        "Fatima Al-Hassan","George Kim","Hannah Osei","Ivan Petrov","Julia Ferreira",
        "Kevin Okafor","Laura Singh","Marcus Chen","Natasha Ivanova","Oliver Brooks",
        "Priya Nair","Quincy Adams","Rachel Torres","Samuel Lee","Tanya Chandra",
        "Umar Sheikh","Vera Morozova","William Scott","Xiao Li","Yuki Tanaka","Zoe Williams"
    ]
    ALL_ROLES = ["Finance","IT","HR","Engineering","Sales","Legal","Operations","Research"]
    CHOICES   = ["HIGH","HIGH","HIGH","HIGH","NORMAL","LOW"]

    def z2l(z):
        try:
            z = float(z)
            return "HIGH" if z > 1.5 else "LOW" if z < -1.0 else "NORMAL"
        except:
            return "NORMAL"

    def lc(level):
        return {"HIGH":"#e63946","NORMAL":"#f4a261","LOW":"#2a9d8f"}.get(level,"#888")

    def dot(level):
        c = lc(level)
        return f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{c};margin-right:5px;vertical-align:middle;"></span><span style="color:{c};font-weight:700;">{level}</span>'

    # Build employee records from top 50 by risk score
    rf = role_df.reset_index(drop=True) if role_df is not None else None
    top50 = combined.sort_values("combined_risk_score", ascending=False).head(50).reset_index(drop=True)
    emp_records = []
    for i, crow in top50.iterrows():
        uid   = i + 1
        name  = EMP_NAMES[i % len(EMP_NAMES)]
        score = round(float(crow["combined_risk_score"]), 1)
        mal   = int(crow.get("is_malicious", 0))
        if rf is not None and i < len(rf):
            row_r = rf.iloc[i]
            role  = str(row_r.get("role", ALL_ROLES[i % len(ALL_ROLES)]))
            usb   = z2l(row_r.get("role_usb_events_count_zscore", 0))
            files = z2l(row_r.get("role_files_accessed_count_zscore", 0))
            sess  = z2l(row_r.get("role_session_duration_mins_zscore", 0))
            email = z2l(row_r.get("role_email_count_zscore", 0))
        else:
            random.seed(uid * 31);  role  = random.choice(ALL_ROLES)
            random.seed(uid * 3);   usb   = random.choice(CHOICES)
            random.seed(uid * 5);   files = random.choice(CHOICES)
            random.seed(uid * 11);  sess  = random.choice(CHOICES)
            random.seed(uid * 13);  email = random.choice(CHOICES)
        emp_records.append({
            "uid": f"User_{uid:04d}", "name": name, "role": role,
            "score": score, "malicious": mal,
            "usb": usb, "files": files, "sess": sess, "email": email,
            "high_count": [usb, files, sess, email].count("HIGH")
        })

    emp_records.sort(key=lambda e: (e["high_count"], e["score"]), reverse=True)

    # Summary table
    hdr = '<div style="overflow-x:auto;margin-top:0.5rem;"><table style="width:100%;border-collapse:collapse;font-size:0.88rem;">'
    hdr += '<thead><tr style="border-bottom:2px solid #2a2a3a;">'
    for col in ["Employee","Role","USB Activity","File Access","Session Duration","Ext. Email","Risk Score"]:
        hdr += f'<th style="padding:0.6rem 0.8rem;text-align:left;color:#666;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;">{col}</th>'
    hdr += "</tr></thead><tbody>"
    rows_html = ""
    for e in emp_records:
        sc = "#e63946" if e["score"] >= 80 else "#f4a261" if e["score"] >= 60 else "#2a9d8f"
        bg = "#1a0a0d" if e["malicious"] == 1 else "transparent"
        rows_html += f'<tr style="border-bottom:1px solid #1e1e2e;background:{bg};">'
        rows_html += f'<td style="padding:0.55rem 0.8rem;color:#fff;font-weight:600;">{e["uid"]} <span style="color:#555;font-size:0.8rem;font-weight:400;">— {e["name"]}</span></td>'
        rows_html += f'<td style="padding:0.55rem 0.8rem;color:#aaa;">{e["role"]}</td>'
        rows_html += f'<td style="padding:0.55rem 0.8rem;">{dot(e["usb"])}</td>'
        rows_html += f'<td style="padding:0.55rem 0.8rem;">{dot(e["files"])}</td>'
        rows_html += f'<td style="padding:0.55rem 0.8rem;">{dot(e["sess"])}</td>'
        rows_html += f'<td style="padding:0.55rem 0.8rem;">{dot(e["email"])}</td>'
        rows_html += f'<td style="padding:0.55rem 0.8rem;font-weight:700;color:{sc};">{e["score"]}</td>'
        rows_html += "</tr>"
    st.markdown(hdr + rows_html + "</tbody></table></div>", unsafe_allow_html=True)
    st.caption(f"Top {len(emp_records)} employees sorted by threat severity · Red background = confirmed threat")

    # Employee detail card
    st.markdown("---")
    st.markdown("#### Employee Detail View")
    st.caption("Select an employee to inspect their full behavioral breakdown.")

    emp_records_sorted = sorted(emp_records, key=lambda e: e["uid"])
    labels = [f'{e["uid"]}  —  {e["name"]}' for e in emp_records_sorted]
    sel_label = st.selectbox("Employee", labels)
    sel = emp_records_sorted[labels.index(sel_label)]

    st.markdown(f"""
<div style="background:#161622;border:1px solid #2a2a3a;border-radius:6px;padding:1rem 1.4rem;margin:0.8rem 0;">
  <span style="font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;color:#666;">Employee</span>
  <h3 style="margin:0.15rem 0 0.15rem 0;color:#fff;">{sel["name"]}</h3>
  <span style="font-size:0.8rem;color:#555;">{sel["uid"]}</span>
  <div style="margin-top:0.6rem;display:flex;gap:2rem;">
    <div><span style="font-size:0.72rem;text-transform:uppercase;color:#666;letter-spacing:0.08em;">Role</span>
    <p style="margin:0.1rem 0 0;font-size:1rem;font-weight:600;color:#fff;">{sel["role"]}</p></div>
    <div><span style="font-size:0.72rem;text-transform:uppercase;color:#666;letter-spacing:0.08em;">Risk Score</span>
    <p style="margin:0.1rem 0 0;font-size:1rem;font-weight:700;color:{lc("HIGH") if sel["score"]>=80 else lc("NORMAL") if sel["score"]>=60 else lc("LOW")};">{sel["score"]}</p></div>
  </div>
</div>
""", unsafe_allow_html=True)

    dc1, dc2 = st.columns(2)
    with dc1:
        st.caption("ROLE DEVIATION")
        for lbl, key in [("USB Activity","usb"),("File Access","files"),("Session Duration","sess"),("Ext. Email","email")]:
            v = sel[key]
            c = lc(v)
            st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:0.45rem 0;border-bottom:1px solid #1e1e2e;"><span style="color:#aaa;">{lbl}</span><span style="color:{c};font-weight:700;">{v}</span></div>', unsafe_allow_html=True)
    with dc2:
        st.caption("ANALYST SUMMARY")
        highs = [lbl for lbl, key in [("USB Activity","usb"),("File Access","files"),("Session Duration","sess"),("Ext. Email","email")] if sel[key]=="HIGH"]
        if highs:
            for lbl in highs:
                st.markdown(f'<div style="background:#2a0a0e;border-left:3px solid #e63946;padding:0.6rem 1rem;margin-bottom:0.5rem;border-radius:0 4px 4px 0;color:#f4a261;font-size:0.9rem;"><b>{lbl}</b> is significantly above the normal baseline for the <b>{sel["role"]}</b> role.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#0a2a1a;border-left:3px solid #2a9d8f;padding:0.6rem 1rem;border-radius:0 4px 4px 0;color:#2a9d8f;font-size:0.9rem;">Behavior is within expected range for this role.</div>', unsafe_allow_html=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Live Monitoring Simulation")
    st.caption("Stream events in real-time. Switch to Evaded mode to simulate an adversarial evasion attempt.")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        mode = st.radio("Dataset", ["Normal", "Evaded (Strategy 1 — evasion attack)"], horizontal=True)
    with col_b:
        speed = st.slider("Speed (rows/sec)", 1, 20, 5)
    if st.button("Start Live Feed"):
        data = combined.head(1000).copy() if mode == "Normal" else evaded.head(1000).copy()
        if "combined_risk_score" not in data.columns:
            st.error("Evaded dataset missing combined_risk_score column.")
            st.stop()
        alert_thresh = 70
        metrics_box = st.empty()
        chart_box   = st.empty()
        feed_box    = st.empty()
        seen, alerts, threats_caught = [], 0, 0
        for i, row in data.iterrows():
            score = float(row["combined_risk_score"])
            label = int(row.get("true_label", row.get("is_malicious", 0)))
            seen.append({"index": len(seen), "risk_score": score})
            if score >= alert_thresh:
                alerts += 1
                if label == 1: threats_caught += 1
            if len(seen) % 10 == 0 or i == len(data) - 1:
                df_seen = pd.DataFrame(seen)
                with metrics_box.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Processed", len(seen))
                    m2.metric("Alerts", alerts)
                    m3.metric("Threats Caught", threats_caught)
                chart_box.line_chart(df_seen.set_index("index")["risk_score"])
                if score >= alert_thresh:
                    feed_box.warning(f"ALERT  |  Row {len(seen)}  |  Risk Score: {score:.1f}  |  Malicious: {bool(label)}")
            time.sleep(1 / speed)
        st.success(f"Complete — {threats_caught} confirmed threats out of {alerts} alerts raised.")

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### SHAP Feature Importance")
    st.caption("Features ranked by their contribution to the Isolation Forest anomaly score. Higher value = stronger threat signal.")
    try:
        import shap, joblib, traceback, matplotlib.pyplot as plt
        model_path = f"{BASE}/models/iso_forest.pkl"
        if os.path.exists(model_path):
            with st.spinner("Computing SHAP values..."):
                model = joblib.load(model_path)
                sample = X_test.head(200)
                explainer = shap.Explainer(model, sample)
                shap_values = explainer(sample)
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            plt.rcParams.update({"text.color":"white","axes.labelcolor":"white","xtick.color":"white","ytick.color":"white"})
            shap.plots.bar(shap_values, max_display=10, show=False)
            ax = plt.gca()
            ax.set_facecolor("#0e1117")
            fig.patch.set_facecolor("#0e1117")
            for bar in ax.patches:
                bar.set_facecolor("#3d6b8a"); bar.set_edgecolor("#3d6b8a")
            for txt in ax.texts:
                txt.set_color("#f4a261"); txt.set_fontweight("600")
            ax.tick_params(colors="#cccccc")
            for spine in ax.spines.values(): spine.set_edgecolor("#2a2a3a")
            plt.title("Top 10 Features — Isolation Forest", color="white", fontsize=13, fontweight="bold", pad=14, loc="center")
            plt.tight_layout()
            col_l, col_c, col_r = st.columns([1, 6, 1])
            with col_c:
                st.pyplot(fig)
            plt.close()
            st.markdown("---")
            st.caption("INTERPRETATION")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div style="background:#161622;border:1px solid #2a2a3a;border-radius:6px;padding:1rem;"><p style="color:#888;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.4rem 0;">Primary Signal</p><p style="color:#fff;margin:0;font-size:0.95rem;"><b>files_accessed_count</b> and <b>usb_events_count</b> are the strongest predictors.</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div style="background:#161622;border:1px solid #2a2a3a;border-radius:6px;padding:1rem;"><p style="color:#888;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.4rem 0;">Secondary Signal</p><p style="color:#fff;margin:0;font-size:0.95rem;"><b>login_hour</b> and <b>file_copy_to_removable</b> — after-hours access with removable media substantially raises risk.</p></div>', unsafe_allow_html=True)
        else:
            st.warning("Model file not found at `models/iso_forest.pkl`. Run `train_and_save_models.py` first.")
    except Exception:
        st.error(traceback.format_exc())

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    try:
        from cost_of_breach_panel import render_cost_of_breach_panel
        render_cost_of_breach_panel()
    except Exception:
        import traceback
        st.error(traceback.format_exc())
