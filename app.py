import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
import plotly.graph_objects as go

# All functions from dashboard.py are copied here for global availability
# (Per requirement: "All functions defined in the notebook will be placed at the top of the Streamlit script (app.py) for global availability.")

@st.cache_data(ttl="2h")
def generate_synthetic_classification_data(num_samples, num_features, n_informative, n_redundant, n_clusters_per_class, random_state, mean_shift=0, std_factor=1, concept_drift_factor=0, performance_degradation_factor=0):
    X, y = make_classification(
        n_samples=num_samples, n_features=num_features, n_informative=n_informative,
        n_redundant=n_redundant, n_clusters_per_class=n_clusters_per_class, random_state=random_state
    )

    for i in range(min(2, num_features)):
        X[:, i] = (X[:, i] * std_factor) + mean_shift

    if concept_drift_factor > 0:
        drift_effect = (X[:, 0] > np.median(X[:, 0])).astype(int)
        flip_indices = np.random.choice(np.where(drift_effect == 1)[0], size=int(num_samples * concept_drift_factor), replace=False)
        y[flip_indices] = 1 - y[flip_indices]

    if performance_degradation_factor > 0:
        num_mislabels = int(num_samples * performance_degradation_factor)
        mislab_indices = np.random.choice(num_samples, size=num_mislabels, replace=False)
        y[mislab_indices] = 1 - y[mislab_indices]

    return X, y

@st.cache_resource
def train_logistic_regression_model(X_train, y_train, random_state=42):
    model = LogisticRegression(random_state=random_state, solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)
    return model

@st.cache_data(ttl="2h")
def predict_with_model(model, X_data):
    return model.predict(X_data)

@st.cache_data(ttl="2h")
def calculate_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

@st.cache_data(ttl="2h")
def calculate_data_drift_ks(baseline_feature_data, current_feature_data):
    baseline_feature_data = np.asarray(baseline_feature_data)
    current_feature_data = np.asarray(current_feature_data)
    baseline_feature_data = baseline_feature_data[~np.isnan(baseline_feature_data)]
    current_feature_data = current_feature_data[~np.isnan(current_feature_data)]
    if len(baseline_feature_data) == 0 or len(current_feature_data) == 0:
        return 1.0, 1.0
    ks_statistic, p_value = ks_2samp(baseline_feature_data, current_feature_data)
    return ks_statistic, p_value

@st.cache_data(ttl="2h")
def calculate_data_drift_jsd(baseline_feature_data, current_feature_data, num_bins=50):
    baseline_feature_data = np.asarray(baseline_feature_data)
    current_feature_data = np.asarray(current_feature_data)
    baseline_feature_data = baseline_feature_data[~np.isnan(baseline_feature_data)]
    current_feature_data = current_feature_data[~np.isnan(current_feature_data)]
    if len(baseline_feature_data) == 0 or len(current_feature_data) == 0:
        return 1.0

    all_data = np.concatenate([baseline_feature_data, current_feature_data])
    min_val, max_val = np.min(all_data), np.max(all_data)
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5
    bins = np.linspace(min_val, max_val, num_bins + 1)

    hist_baseline, _ = np.histogram(baseline_feature_data, bins=bins, density=True)
    hist_current, _ = np.histogram(current_feature_data, bins=bins, density=True)

    epsilon = 1e-10
    hist_baseline = hist_baseline + epsilon
    hist_current = hist_current + epsilon

    jsd = jensenshannon(hist_baseline, hist_current)
    return jsd**2

def run_monitoring_step(champion_model, historical_data_X, historical_data_y, baseline_X, baseline_y, time_step, drift_params, alert_thresholds):
    batch_num_samples = 100
    current_batch_X, current_batch_y = generate_synthetic_classification_data(
        num_samples=batch_num_samples, num_features=baseline_X.shape[1], n_informative=3,
        n_redundant=0, n_clusters_per_class=1, random_state=42 + time_step,
        mean_shift=drift_params.get('mean_shift', 0),
        std_factor=drift_params.get('std_factor', 1),
        concept_drift_factor=drift_params.get('concept_drift_factor', 0),
        performance_degradation_factor=drift_params.get('performance_degradation_factor', 0)
    )
    current_batch_df = pd.DataFrame(current_batch_X, columns=[f'feature_{i}' for i in range(baseline_X.shape[1])])

    if historical_data_X.empty:
        monitoring_data_X = current_batch_df
        monitoring_data_y = current_batch_y
    else:
        monitoring_data_X = pd.concat([historical_data_X, current_batch_df], ignore_index=True)
        monitoring_data_y = np.concatenate([historical_data_y, current_batch_y])

    y_pred_current = predict_with_model(champion_model, current_batch_X)
    performance_metrics = calculate_classification_metrics(current_batch_y, y_pred_current)

    ks_statistic_f0, _ = calculate_data_drift_ks(baseline_X[:, 0], current_batch_X[:, 0])
    jsd_statistic_f0 = calculate_data_drift_jsd(baseline_X[:, 0], current_batch_X[:, 0])

    results = {
        'accuracy': performance_metrics['accuracy'],
        'precision': performance_metrics['precision'],
        'recall': performance_metrics['recall'],
        'f1_score': performance_metrics['f1_score'],
        'ks_statistic_feature_0': ks_statistic_f0,
        'jsd_statistic_feature_0': jsd_statistic_f0,
        'alerts': {},
        'monitoring_data_X': monitoring_data_X,
        'monitoring_data_y': monitoring_data_y
    }

    if performance_metrics['accuracy'] < alert_thresholds['accuracy_min']:
        results['alerts']['accuracy'] = f"Accuracy threshold crossed: {performance_metrics['accuracy']:.4f} (Threshold: {alert_thresholds['accuracy_min']:.4f})"
    if ks_statistic_f0 > alert_thresholds['ks_max']:
        results['alerts']['ks_feature_0'] = f"K-S Stat (Feature 0) threshold crossed: {ks_statistic_f0:.4f} (Threshold: {alert_thresholds['ks_max']:.4f})"
    if jsd_statistic_f0 > alert_thresholds['jsd_max']:
        results['alerts']['jsd_feature_0'] = f"JSD (Feature 0) threshold crossed: {jsd_statistic_f0:.4f} (Threshold: {alert_thresholds['jsd_max']:.4f})"

    return results

def compare_champion_challenger(champion_metrics, challenger_metrics, champion_drift_metrics, challenger_drift_metrics):
    metrics_df = pd.DataFrame({
        'Champion': champion_metrics,
        'Challenger': challenger_metrics
    }).T
    metrics_df.index.name = 'Model'

    fig_perf_comp = go.Figure(data=[
        go.Bar(name='Champion', x=metrics_df.columns, y=metrics_df.loc['Champion'], marker_color='blue', hovertemplate='Model: Champion<br>Metric: %{x}<br>Value: %{y:.4f}<extra></extra>'),
        go.Bar(name='Challenger', x=metrics_df.columns, y=metrics_df.loc['Challenger'], marker_color='red', hovertemplate='Model: Challenger<br>Metric: %{x}<br>Value: %{y:.4f}<extra></extra>')
    ])
    fig_perf_comp.update_layout(barmode='group', title='Performance Metrics: Champion vs. Challenger', yaxis_title='Score')
    st.plotly_chart(fig_perf_comp)

    drift_df = pd.DataFrame({
        'Champion': champion_drift_metrics,
        'Challenger': challenger_drift_metrics
    }).T
    drift_df.index.name = 'Model'

    fig_drift_comp = go.Figure(data=[
        go.Bar(name='Champion', x=drift_df.columns, y=drift_df.loc['Champion'], marker_color='blue', hovertemplate='Model: Champion<br>Metric: %{x}<br>Value: %{y:.4f}<extra></extra>'),
        go.Bar(name='Challenger', x=drift_df.columns, y=drift_df.loc['Challenger'], marker_color='red', hovertemplate='Model: Challenger<br>Metric: %{x}<br>Value: %{y:.4f}<extra></extra>')
    ])
    fig_drift_comp.update_layout(barmode='group', title='Drift Metrics: Champion vs. Challenger (on Recent Data)', yaxis_title='Drift Statistic')
    st.plotly_chart(fig_drift_comp)

def log_manual_review(time_step, alert_details):
    st.info(f"Manual Review Log: Time Step {time_step}, Alert Details: {alert_details}")

def init_session_state():
    if 'baseline_X' not in st.session_state:
        st.session_state.baseline_X = None
        st.session_state.baseline_y = None
        st.session_state.baseline_df = pd.DataFrame()
        st.session_state.baseline_metrics = {}
        st.session_state.champion_model = None
        st.session_state.challenger_model = None
        st.session_state.historical_accuracy = []
        st.session_state.historical_precision = []
        st.session_state.historical_recall = []
        st.session_state.historical_f1 = []
        st.session_state.historical_ks_stats_f0 = []
        st.session_state.historical_jsd_stats_f0 = []
        st.session_state.historical_alerts = []
        st.session_state.monitoring_data_X = pd.DataFrame()
        st.session_state.monitoring_data_y = np.array([])
        st.session_state.current_t_step = 0
        st.session_state.drift_parameters = {
            'mean_shift': 0.0,
            'std_factor': 1.0,
            'concept_drift_factor': 0.0,
            'performance_degradation_factor': 0.0
        }
        st.session_state.alert_thresholds = {
            'accuracy_min': 0.70,
            'ks_max': 0.30,
            'jsd_max': 0.20
        }
        st.session_state.initial_champion_trained = False
        st.session_state.random_state = 42

# Main application starts here
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

st.markdown("""
In this lab, we develop an interactive Streamlit application to simulate and monitor the health of AI models.
This dashboard is designed to provide Risk Managers with a practical understanding of AI Model Risk Management (MRM),
particularly focusing on aspects like continuous monitoring, drift detection, and intervention strategies,
aligning with regulatory guidance such as SR 11-7.

The application demonstrates how to:
- **Continuously Monitor** key performance and data characteristics of an AI model.
- **Detect and Distinguish** between performance degradation, data drift ($P(X)$ changes), and concept drift ($P(Y|X)$ changes).
- **Configure and Respond** to threshold-based alerts for deviations in model health.
- **Simulate Intervention Strategies** including model retraining, Champion-Challenger comparisons, and manual review processes.
- **Reinforce SR 11-7 Principles** for ongoing validation and risk control of adaptive AI systems.

Navigate through the "Dashboard" page to interact with the model health simulator.
""")

page = st.sidebar.selectbox(label="Navigation", options=["Dashboard"])

if page == "Dashboard":
    from application_pages.dashboard import main
    main()


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
