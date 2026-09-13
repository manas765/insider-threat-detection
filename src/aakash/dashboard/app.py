import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Insider Threat Dashboard", layout="wide")
st.title("Insider Threat Detection Dashboard")
st.markdown("---")

BASE = '/Users/aakashsairam/insider-threat-detection'

@st.cache_data
def load_data():
    X_test = pd.read_csv(f'{BASE}/data/processed/X_test.csv')
    y_test = pd.read_csv(f'{BASE}/data/processed/y_test.csv').values.ravel()
    risk_df = pd.read_csv(f'{BASE}/reports/combined_risk_scores.csv')
    if_shap = pd.read_csv(f'{BASE}/reports/iso_forest_explanations.csv')
    svm_shap = pd.read_csv(f'{BASE}/reports/ocsvm_explanations.csv')
    return X_test, y_test, risk_df, if_shap, svm_shap

X_test, y_test, risk_df, if_shap, svm_shap = load_data()

X_test['is_malicious'] = y_test
X_test['risk_score'] = (risk_df['combined_risk_score'].values * 100).round(2)
X_test['autoencoder_score'] = risk_df['autoencoder_scaled'].values.round(4)
X_test['iso_forest_score'] = risk_df['iso_forest_scaled'].values.round(4)
X_test['svm_score'] = risk_df['oc_svm_scaled'].values.round(4)

st.sidebar.header("Filters")
threshold = st.sidebar.slider("Risk score threshold", 0, 100, 20)
show_flagged = st.sidebar.checkbox("Show flagged only", value=False)

flagged = X_test[X_test['risk_score'] >= threshold]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total records", len(X_test))
col2.metric("Flagged", len(flagged))
col3.metric("Actual threats", int(y_test.sum()))
col4.metric("Threats caught", int(flagged['is_malicious'].sum()))

st.markdown("---")
st.subheader("Top flagged cases")
top_flagged = X_test[X_test['is_malicious'] == 1].sort_values('risk_score', ascending=False).head(20)
st.dataframe(top_flagged[['login_hour','after_hours_flag','session_duration_mins','usb_events_count','files_accessed_count','email_count','unique_domains_visited','email_ext_recipient_count','risk_score','autoencoder_score','iso_forest_score','svm_score']].round(3), width='stretch')

st.markdown("---")
st.subheader("Why was it flagged? — Top SHAP features (Isolation Forest)")
st.dataframe(if_shap.head(20), width='stretch')

st.markdown("---")
st.subheader("User activity log")
display_df = flagged if show_flagged else X_test
st.dataframe(display_df[['login_hour','after_hours_flag','session_duration_mins','usb_events_count','files_accessed_count','email_count','unique_domains_visited','email_ext_recipient_count','risk_score','is_malicious']].round(2), width='stretch')

st.markdown("---")
st.subheader("Risk score distribution")
col_a, col_b = st.columns(2)
with col_a:
    st.write("Normal records")
    st.bar_chart(pd.DataFrame({'risk_score': X_test[X_test['is_malicious'] == 0]['risk_score'].sample(500)}))
with col_b:
    st.write("Malicious records")
    st.bar_chart(pd.DataFrame({'risk_score': X_test[X_test['is_malicious'] == 1]['risk_score']}))

st.markdown("---")
st.subheader("Alert fatigue simulation")
thresholds = range(0, 101, 5)
workload = []
recall_list = []
for t in thresholds:
    flagged_t = X_test[X_test['risk_score'] >= t]
    caught = int(flagged_t['is_malicious'].sum())
    total_threats = int(y_test.sum())
    workload.append(len(flagged_t))
    recall_list.append(round(caught / total_threats * 100, 1) if total_threats > 0 else 0)

fatigue_df = pd.DataFrame({'threshold': list(thresholds), 'alerts_to_review': workload, 'threats_caught_pct': recall_list})
col_c, col_d = st.columns(2)
with col_c:
    st.write("Alerts to review per day")
    st.line_chart(fatigue_df.set_index('threshold')['alerts_to_review'])
with col_d:
    st.write("% of threats caught")
    st.line_chart(fatigue_df.set_index('threshold')['threats_caught_pct'])
