import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Live Threat Monitor", layout="wide")
st.title("Live Insider Threat Monitor")

BASE = '/Users/aakashsairam/insider-threat-detection'

risk_df = pd.read_csv(f'{BASE}/reports/combined_risk_scores.csv', nrows=1000)
risk_df['risk_score'] = risk_df['combined_risk_score'].round(2)

st.success(f"Loaded {len(risk_df)} events")

threshold = st.sidebar.slider("Alert threshold", 0.0, float(risk_df['risk_score'].max()), float(risk_df['risk_score'].quantile(0.95)))
start = st.sidebar.button("Start monitoring")

if start:
    alerts = []
    placeholder = st.empty()
    
    for i in range(len(risk_df)):
        row = risk_df.iloc[i]
        
        if row['risk_score'] >= threshold:
            alerts.append({
                'event': i+1,
                'risk_score': row['risk_score'],
                'iso_forest': round(row['iso_forest_scaled'], 3),
                'oc_svm': round(row['oc_svm_scaled'], 3),
                'autoencoder': round(row['autoencoder_scaled'], 3),
                'actual_threat': int(row['true_label'])
            })
        
        with placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Events seen", i+1)
            col2.metric("Alerts raised", len(alerts))
            col3.metric("Threats caught", sum(a['actual_threat'] for a in alerts))
            col4.metric("Miss rate", f"{round((1 - sum(a['actual_threat'] for a in alerts)/max(1, int(risk_df['true_label'][:i+1].sum())))*100, 1)}%")
            
            if alerts:
                st.dataframe(pd.DataFrame(alerts[-10:]))
        
        time.sleep(0.05)
    
    st.balloons()
    st.success(f"Done — processed {len(risk_df)} events, caught {sum(a['actual_threat'] for a in alerts)} threats.")
