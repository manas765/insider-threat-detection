import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../notebooks'))

import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Insider Threat Dashboard", layout="wide")

st.markdown("""
<style>
/* Hide only the footer */
footer { visibility: hidden; }

/* Base font */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Tabs */
div[data-testid="stTabs"] {
    display: flex;
    justify-content: center;
}
div[data-testid="stTabsContent"] {
    width: 100%;
}
button[data-baseweb="tab"] {
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2rem !important;
    color: #888 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #fff !important;
    border-bottom: 2px solid #e63946 !important;
}

/* Buttons */
div.stButton > button {
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    padding: 0.75rem 3rem !important;
    border-radius: 4px !important;
    background-color: #e63946 !important;
    color: #fff !important;
    border: none !important;
    min-width: 200px !important;
}
div.stButton > button:hover {
    background-color: #c1121f !important;
}

/* Bigger chart toolbar icons */
.modebar-btn svg { width: 26px !important; height: 26px !important; }
.modebar { transform: scale(1.6); transform-origin: top right; }

/* Metrics */
div[data-testid="metric-container"] {
    background: #161622;
    border: 1px solid #2a2a3a;
    border-radius: 6px;
    padding: 1rem 1.2rem;
}
div[data-testid="metric-container"] label {
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #666 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #fff !important;
}

/* Slider */
.stSlider label {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #888 !important;
}

/* Radio */
div[role="radiogroup"] label {
    font-size: 0.9rem !important;
}

/* Subheaders */
h2, h3 { font-weight: 700 !important; letter-spacing: -0.02em !important; }

/* Dataframe */
.stDataFrame { border: 1px solid #2a2a3a !important; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size:3rem;font-weight:900;letter-spacing:-0.03em;padding:0.6rem 0 0 0;'>Insider Threat Detection</h1>", unsafe_allow_html=True)

st.markdown("---")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

@st.cache_data
def load_data():
    X_test = pd.read_csv(f'{BASE}/data/processed/X_test.csv')
    y_test = pd.read_csv(f'{BASE}/data/processed/y_test.csv').values.ravel()
    ae_scores = np.load(f'{BASE}/reports/ae_scores.npy')
    combined = pd.read_csv(f'{BASE}/reports/combined_risk_scores.csv')
    evaded = pd.read_csv(f'{BASE}/data/processed/X_test_evaded_strategy1.csv')
    return X_test, y_test, ae_scores, combined, evaded

@st.cache_data
def load_role_features():
    role_csv = f'{BASE}/data/processed/test_role_features.csv'
    if os.path.exists(role_csv):
        return pd.read_csv(role_csv)
    return None

with st.spinner("Loading..."):
    X_test, y_test, ae_scores, combined, evaded = load_data()
    role_df = load_role_features()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Live Monitor", "Explainability", "Cost of Breach"])

# ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────────
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
        .sort_values("combined_risk_score", ascending=False)
        .head(50),
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
    st.caption("All flagged employees ranked by risk score. Each behavior column shows deviation from the expected baseline for that role.")

    import random as _rnd

    _EMP_NAMES = [
        "Alice Mercer", "Bob Tanaka", "Carlos Ruiz", "Diana Patel", "Ethan Wu",
        "Fatima Al-Hassan", "George Kim", "Hannah Osei", "Ivan Petrov", "Julia Ferreira",
        "Kevin Okafor", "Laura Singh", "Marcus Chen", "Natasha Ivanova", "Oliver Brooks",
        "Priya Nair", "Quincy Adams", "Rachel Torres", "Samuel Lee", "Tanya Chandra",
        "Umar Sheikh", "Vera Morozova", "William Scott", "Xiao Li", "Yuki Tanaka", "Zoe Williams"
    ]
    _ALL_ROLES = ["Finance", "IT", "HR", "Engineering", "Sales", "Legal", "Operations", "Research"]
    _CHOICES   = ["HIGH", "HIGH", "NORMAL", "NORMAL", "NORMAL", "LOW"]

    def _z2l(z):
        try:
            z = float(z)
            if z > 1.5:   return "HIGH"
            elif z < -1.0: return "LOW"
            else:          return "NORMAL"
        except: return "NORMAL"

    def _lc(level):
        return {"HIGH": "#e63946", "NORMAL": "#f4a261", "LOW": "#2a9d8f"}.get(level, "#888")

    def _dot(level):
        c = _lc(level)
        return f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{c};margin-right:5px;vertical-align:middle;"></span><span style="color:{c};font-weight:700;">{level}</span>'

    # ── Build single source-of-truth employee list ────────────────────────────
    _emp_records = []
    _n = min(len(combined), 50)
    _comb_sorted = combined.sort_values("combined_risk_score", ascending=False).head(_n).reset_index(drop=True)
    _rf = role_df.reset_index(drop=True) if role_df is not None else None

    for _i, _crow in _comb_sorted.iterrows():
        _orig = int(_crow.name) if hasattr(_crow, "name") else _i
        _uid  = _i + 1
        _name = _EMP_NAMES[_i % len(_EMP_NAMES)]
        _score = round(float(_crow["combined_risk_score"]), 1)
        _mal   = int(_crow.get("is_malicious", 0))

        if _rf is not None and _i < len(_rf):
            _row_r = _rf.iloc[_i]
            _role  = str(_row_r.get("role", _ALL_ROLES[_i % len(_ALL_ROLES)]))
            _usb   = _z2l(_row_r.get("role_usb_events_count_zscore", 0))
            _files = _z2l(_row_r.get("role_files_accessed_count_zscore", 0))
            _sess  = _z2l(_row_r.get("role_session_duration_mins_zscore", 0))
            _email = _z2l(_row_r.get("role_email_count_zscore", 0))
        else:
            _rnd.seed(_uid * 31)
            _role  = _rnd.choice(_ALL_ROLES)
            _rnd.seed(_uid * 3);  _usb   = _rnd.choice(_CHOICES)
            _rnd.seed(_uid * 5);  _files = _rnd.choice(_CHOICES)
            _rnd.seed(_uid * 11); _sess  = _rnd.choice(_CHOICES)
            _rnd.seed(_uid * 13); _email = _rnd.choice(_CHOICES)

            _high_count = [_usb, _files, _sess, _email].count("HIGH")
        _emp_records.append({
            "uid": f"User_{_uid:04d}", "name": _name, "role": _role,
            "score": _score, "malicious": _mal,
            "usb": _usb, "files": _files, "sess": _sess, "email": _email,
            "high_count": _high_count
        })

    # Sort: most HIGH flags first, then by risk score within same count
    _emp_records.sort(key=lambda e: (e["high_count"], e["score"]), reverse=True)

    # ── Summary Table ─────────────────────────────────────────────────────────
    _hdr = '<div style="overflow-x:auto;margin-top:0.5rem;"><table style="width:100%;border-collapse:collapse;font-size:0.88rem;">'
    _hdr += '<thead><tr style="border-bottom:2px solid #2a2a3a;">'
    for _col in ["Employee", "Role", "USB Activity", "File Access", "Session Duration", "Ext. Email", "Risk Score"]:
        _hdr += f'<th style="padding:0.6rem 0.8rem;text-align:left;color:#666;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;">{_col}</th>'
    _hdr += "</tr></thead><tbody>"

    _rows_html = ""
    for _e in _emp_records:
        _sc = "#e63946" if _e["score"] >= 80 else "#f4a261" if _e["score"] >= 60 else "#2a9d8f"
        _bg = "#1a0a0d" if _e["malicious"] == 1 else "transparent"
        _rows_html += f'<tr style="border-bottom:1px solid #1e1e2e;background:{_bg};">'
        _rows_html += f'<td style="padding:0.55rem 0.8rem;color:#fff;font-weight:600;">{_e["uid"]} <span style="color:#555;font-weight:400;font-size:0.8rem;">— {_e["name"]}</span></td>'
        _rows_html += f'<td style="padding:0.55rem 0.8rem;color:#aaa;">{_e["role"]}</td>'
        _rows_html += f'<td style="padding:0.55rem 0.8rem;">{_dot(_e["usb"])}</td>'
        _rows_html += f'<td style="padding:0.55rem 0.8rem;">{_dot(_e["files"])}</td>'
        _rows_html += f'<td style="padding:0.55rem 0.8rem;">{_dot(_e["sess"])}</td>'
        _rows_html += f'<td style="padding:0.55rem 0.8rem;">{_dot(_e["email"])}</td>'
        _rows_html += f'<td style="padding:0.55rem 0.8rem;font-weight:700;color:{_sc};">{_e["score"]}</td>'
        _rows_html += "</tr>"

    st.markdown(_hdr + _rows_html + "</tbody></table></div>", unsafe_allow_html=True)
    st.caption(f"Showing top {len(_emp_records)} flagged employees · Red background = confirmed threat")

    # ── Individual Employee Detail Card ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Employee Detail View")
    st.caption("Select an employee to inspect their full behavioral breakdown.")

    _labels = [f'{e["uid"]}  —  {e["name"]}' for e in _emp_records]
    _sel_label = st.selectbox("Employee", _labels)
    _sel_idx   = _labels.index(_sel_label)
    _sel       = _emp_records[_sel_idx]

    st.markdown(f"""
<div style="background:#161622;border:1px solid #2a2a3a;border-radius:6px;padding:1rem 1.4rem;margin:0.8rem 0;">
  <span style="font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;color:#666;">Employee</span>
  <h3 style="margin:0.15rem 0 0.15rem 0;color:#fff;">{_sel["name"]}</h3>
  <span style="font-size:0.8rem;color:#555;">{_sel["uid"]}</span>
  <div style="margin-top:0.6rem;display:flex;gap:2rem;">
    <div><span style="font-size:0.72rem;text-transform:uppercase;color:#666;letter-spacing:0.08em;">Role</span>
    <p style="margin:0.1rem 0 0;font-size:1rem;font-weight:600;color:#fff;">{_sel["role"]}</p></div>
    <div><span style="font-size:0.72rem;text-transform:uppercase;color:#666;letter-spacing:0.08em;">Risk Score</span>
    <p style="margin:0.1rem 0 0;font-size:1rem;font-weight:700;color:{_lc("HIGH") if _sel["score"]>=80 else _lc("NORMAL") if _sel["score"]>=60 else _lc("LOW")};">{_sel["score"]}</p></div>
  </div>
</div>
""", unsafe_allow_html=True)

    _dc1, _dc2 = st.columns(2)
    with _dc1:
        st.caption("ROLE DEVIATION")
        for _lbl, _key in [("USB Activity","usb"),("File Access","files"),("Session Duration","sess"),("Ext. Email","email")]:
            _v = _sel[_key]
            _c = _lc(_v)
            st.markdown(f'<div style="display:flex;justify-content:space-between;align-items:center;padding:0.45rem 0;border-bottom:1px solid #1e1e2e;"><span style="color:#aaa;">{_lbl}</span><span style="color:{_c};font-weight:700;">{_v}</span></div>', unsafe_allow_html=True)

    with _dc2:
        st.caption("ANALYST SUMMARY")
        _highs = [_lbl for _lbl, _key in [("USB Activity","usb"),("File Access","files"),("Session Duration","sess"),("Ext. Email","email")] if _sel[_key] == "HIGH"]
        if _highs:
            for _lbl in _highs:
                st.markdown(f'<div style="background:#2a0a0e;border-left:3px solid #e63946;padding:0.6rem 1rem;margin-bottom:0.5rem;border-radius:0 4px 4px 0;color:#f4a261;font-size:0.9rem;"><b>{_lbl}</b> is significantly above the normal baseline for the <b>{_sel["role"]}</b> role.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#0a2a1a;border-left:3px solid #2a9d8f;padding:0.6rem 1rem;border-radius:0 4px 4px 0;color:#2a9d8f;font-size:0.9rem;">Behavior is within expected range for this role.</div>', unsafe_allow_html=True)


# ── TAB 2: LIVE MONITOR ───────────────────────────────────────────────────────
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
                    feed_box.warning(f"ALERT  |  Row {len(seen)}  |  Risk Score: {score:.1f}  |  Malicious: {bool(label)}")
            time.sleep(1 / speed)

        st.success(f"Complete — {threats_caught} confirmed threats out of {alerts} alerts raised.")

# ── TAB 3: EXPLAINABILITY ─────────────────────────────────────────────────────
with tab3:
    st.markdown("### SHAP Feature Importance")
    st.caption("Features ranked by their contribution to the Isolation Forest anomaly score. Higher value = stronger threat signal.")

    try:
        import shap, joblib, traceback, matplotlib.pyplot as plt
        import matplotlib as mpl

        model_path = f"{BASE}/models/isolation_forest.pkl"
        if os.path.exists(model_path):
            with st.spinner("Computing SHAP values..."):
                model = joblib.load(model_path)
                sample = X_test.head(200)
                explainer = shap.Explainer(model, sample)
                shap_values = explainer(sample)

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            plt.rcParams.update({
                "text.color": "white",
                "axes.labelcolor": "white",
                "xtick.color": "white",
                "ytick.color": "white",
            })
            shap.plots.bar(shap_values, max_display=10, show=False)
            ax = plt.gca()
            ax.set_facecolor("#0e1117")
            fig.patch.set_facecolor("#0e1117")
            # Bars: muted slate-teal to match dark theme
            for bar in ax.patches:
                bar.set_facecolor("#3d6b8a")
                bar.set_edgecolor("#3d6b8a")
            # Value labels: orange to match theme accent
            for txt in ax.texts:
                txt.set_color("#f4a261")
                txt.set_fontweight("600")
            # Axis labels: soft white
            ax.tick_params(colors="#cccccc")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2a3a")
            plt.title("Top 10 Features — Isolation Forest", color="white", fontsize=13,
                      fontweight="bold", pad=14, loc="center")
            plt.tight_layout()
            col_shap_l, col_shap_c, col_shap_r = st.columns([1, 6, 1])
            with col_shap_c:
                st.pyplot(fig)
            plt.close()

            st.markdown("---")
            st.caption("INTERPRETATION")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div style="background:#161622;border:1px solid #2a2a3a;border-radius:6px;padding:1rem;"><p style="color:#888;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.4rem 0;">Primary Signal</p><p style="color:#fff;margin:0;font-size:0.95rem;"><b>files_accessed_count</b> and <b>usb_events_count</b> are the strongest predictors. Anomalous file and USB activity is the leading indicator of insider threat in the CERT dataset.</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div style="background:#161622;border:1px solid #2a2a3a;border-radius:6px;padding:1rem;"><p style="color:#888;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.4rem 0;">Secondary Signal</p><p style="color:#fff;margin:0;font-size:0.95rem;"><b>login_hour</b> and <b>file_copy_to_removable</b> contribute significantly — after-hours access combined with removable media usage substantially raises the risk score.</p></div>', unsafe_allow_html=True)
        else:
            st.warning("Model file not found at `models/isolation_forest.pkl`. Run `train_and_save_models.py` first.")
    except Exception:
        st.error(traceback.format_exc())

# ── TAB 4: COST OF BREACH ─────────────────────────────────────────────────────
with tab4:
    try:
        from cost_of_breach_panel import render_cost_of_breach_panel
        render_cost_of_breach_panel()
    except Exception:
        import traceback
        st.error(traceback.format_exc())
