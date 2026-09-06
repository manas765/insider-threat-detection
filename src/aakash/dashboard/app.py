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
    ae_scores = np.load(f'{BASE}/reports/ae_scores.npy')
    pushkar_df = pd.read_csv(f'{BASE}/reports/model_scores_for_comparison.csv')
    return X_test, y_test, ae_scores, pushkar_df

X_test, y_test, ae_scores, pushkar_df = load_data()

# Add labels and scores
X_test['is_malicious'] = y_test
X_test['autoencoder_score'] = ae_scores
if_scores = pushkar_df['iso_forest_score'].values
svm_scores = pushkar_df['oc_svm_score'].values
X_test['iso_forest_score'] = if_scores
X_test['svm_score'] = svm_scores

# Normalize all scores to 0-1
ae_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min())
if_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
svm_norm = (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min())

# Weighted ensemble risk score 0-100
X_test['risk_score'] = (0.4 * ae_norm + 0.4 * if_norm + 0.2 * svm_norm) * 100

# Sidebar
st.sidebar.header("Filters")
threshold = st.sidebar.slider("Risk score threshold", 0, 100, 50)
show_flagged = st.sidebar.checkbox("Show flagged only", value=False)

# Metrics
flagged = X_test[X_test['risk_score'] >= threshold]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total records", len(X_test))
col2.metric("Flagged", len(flagged))
col3.metric("Actual threats", int(y_test.sum()))
col4.metric("Threats caught", int(flagged['is_malicious'].sum()))

st.markdown("---")

# Top flagged threats
st.subheader("Top flagged cases")
top_flagged = X_test[X_test['is_malicious'] == 1].sort_values('risk_score', ascending=False).head(20)
st.dataframe(
    top_flagged[['login_hour', 'after_hours_flag', 'session_duration_mins',
                 'usb_events_count', 'files_accessed_count', 'email_count',
                 'risk_score', 'autoencoder_score', 'iso_forest_score', 'svm_score']].round(3),
    use_container_width=True
)

st.markdown("---")

# Full activity log
st.subheader("User activity log")
display_df = flagged if show_flagged else X_test
st.dataframe(
    display_df[['login_hour', 'after_hours_flag', 'session_duration_mins',
                'usb_events_count', 'files_accessed_count', 'email_count',
                'risk_score', 'is_malicious']].round(2),
    use_container_width=True
)

st.markdown("---")

# Risk score distribution
st.subheader("Risk score distribution")
col_a, col_b = st.columns(2)

with col_a:
    st.write("Normal records")
    st.bar_chart(
        pd.DataFrame({'risk_score': X_test[X_test['is_malicious'] == 0]['risk_score'].sample(500)})
    )

with col_b:
    st.write("Malicious records")
    st.bar_chart(
        pd.DataFrame({'risk_score': X_test[X_test['is_malicious'] == 1]['risk_score']})
    )

st.markdown("---")

# Model comparison
st.subheader("Model score comparison on flagged records")
st.dataframe(
    flagged[['risk_score', 'autoencoder_score', 'iso_forest_score', 'svm_score', 'is_malicious']].round(3),
    use_container_width=True
)