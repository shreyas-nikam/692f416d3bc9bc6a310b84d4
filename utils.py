
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.special import rel_entr
import plotly.graph_objects as go


def generate_synthetic_classification_data(num_samples=1000, num_features=5, n_informative=3, n_redundant=0, n_clusters_per_class=1, random_state=42, mean_shift=0, std_factor=1, concept_drift_factor=0, performance_degradation_factor=0):
    """
    Generates synthetic classification data with optional drift factors.
    - mean_shift: Shifts the mean of features, simulating data drift.
    - std_factor: Multiplies the standard deviation of features, simulating data drift (increased variance).
    - concept_drift_factor: Changes the underlying relationship between features and target, simulating concept drift.
    - performance_degradation_factor: Directly degrades model performance by flipping some labels.
    """
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class,
        random_state=random_state
    )

    # Apply data drift (mean shift and std factor)
    if mean_shift != 0 or std_factor != 1:
        for i in range(X.shape[1]):
            X[:, i] = (X[:, i] * std_factor) + mean_shift

    # Apply concept drift (alter relationship between a feature and target)
    if concept_drift_factor != 0:
        # Example: Flip labels based on a specific feature's value for some samples
        feature_to_drift = 0  # arbitrary choice
        threshold = np.mean(X[:, feature_to_drift])
        drift_indices = np.where(X[:, feature_to_drift] > threshold)[0]
        num_flips = int(len(drift_indices) * concept_drift_factor)
        if num_flips > 0:
            flip_indices_subset = np.random.choice(drift_indices, num_flips, replace=False)
            y[flip_indices_subset] = 1 - y[flip_indices_subset] # Flip the label

    # Apply performance degradation (directly flip labels post-concept drift if any)
    if performance_degradation_factor != 0:
        num_flips = int(num_samples * performance_degradation_factor)
        if num_flips > 0:
            flip_indices = np.random.choice(num_samples, num_flips, replace=False)
            y[flip_indices] = 1 - y[flip_indices]

    return X, y

def train_logistic_regression_model(X_train, y_train, random_state=42):
    """Trains a Logistic Regression model."""
    model = LogisticRegression(random_state=random_state, solver='liblinear')
    model.fit(X_train, y_train)
    return model

def predict_with_model(model, X_data):
    """Generates predictions from a given model."""
    return model.predict(X_data)

def calculate_classification_metrics(y_true, y_pred):
    """Computes Accuracy, Precision, Recall, F1-Score."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

def calculate_data_drift_ks(baseline_feature_data, current_feature_data):
    """Calculates the Kolmogorov-Smirnov (K-S) statistic."""
    if len(baseline_feature_data) == 0 or len(current_feature_data) == 0:
        return 0.0, 1.0 # Return 0 for statistic, 1 for p-value if data is empty
    ks_stat, p_value = ks_2samp(baseline_feature_data, current_feature_data)
    return ks_stat, p_value

def _kl_divergence(p, q):
    """Calculates KL divergence for two probability distributions."""
    return np.sum(rel_entr(p, q))

def calculate_data_drift_jsd(baseline_feature_data, current_feature_data, num_bins=50):
    """Calculates Jensen-Shannon Divergence (JSD)."""
    if len(baseline_feature_data) == 0 or len(current_feature_data) == 0:
        return 0.0

    # Determine common bins for both distributions
    all_data = np.concatenate([baseline_feature_data, current_feature_data])
    min_val, max_val = all_data.min(), all_data.max()
    bins = np.linspace(min_val, max_val, num_bins + 1)

    # Get histograms (counts) and normalize to probabilities
    hist_baseline, _ = np.histogram(baseline_feature_data, bins=bins, density=True)
    hist_current, _ = np.histogram(current_feature_data, bins=bins, density=True)

    # Add a small epsilon to avoid log(0) if any bin is empty
    epsilon = 1e-10
    p = hist_baseline + epsilon
    q = hist_current + epsilon

    # Calculate M = 0.5 * (P + Q)
    m = 0.5 * (p + q)

    # Calculate JSD
    jsd = 0.5 * (_kl_divergence(p, m) + _kl_divergence(q, m))
    return np.sqrt(jsd) # Often reported as square root of JSD


def run_monitoring_step(champion_model, historical_data_X_df, historical_data_y_arr, baseline_X, baseline_y, time_step, drift_params, alert_thresholds, num_samples_per_step=100):
    """
    Simulates one batch of production data, calculates metrics, checks for alerts.
    Returns a dictionary of results including new monitoring data and alert status.
    """
    # Generate new batch of data with applied drift
    current_batch_X, current_batch_y = generate_synthetic_classification_data(
        num_samples=num_samples_per_step, num_features=baseline_X.shape[1],
        random_state=42 + time_step, # Vary random state for each step
        mean_shift=drift_params.get('mean_shift', 0),
        std_factor=drift_params.get('std_factor', 1),
        concept_drift_factor=drift_params.get('concept_drift_factor', 0),
        performance_degradation_factor=drift_params.get('performance_degradation_factor', 0)
    )

    current_batch_X_df = pd.DataFrame(current_batch_X, columns=[f'feature_{i}' for i in range(baseline_X.shape[1])])

    # Accumulate monitoring data
    new_monitoring_data_X_df = pd.concat([historical_data_X_df, current_batch_X_df], ignore_index=True)
    new_monitoring_data_y_arr = np.concatenate([historical_data_y_arr, current_batch_y])

    # Evaluate Champion model on the *latest batch* for performance metrics
    y_pred_current_batch = predict_with_model(champion_model, current_batch_X)
    current_performance_metrics = calculate_classification_metrics(current_batch_y, y_pred_current_batch)

    # Calculate data drift for feature_0 (as specified)
    baseline_feature_0 = pd.DataFrame(baseline_X, columns=[f'feature_{i}' for i in range(baseline_X.shape[1})])['feature_0']
    current_feature_0 = current_batch_X_df['feature_0']

    ks_stat_f0, _ = calculate_data_drift_ks(baseline_feature_0, current_feature_0)
    jsd_stat_f0 = calculate_data_drift_jsd(baseline_feature_0, current_feature_0)

    alerts = {}
    if current_performance_metrics['accuracy'] < alert_thresholds['accuracy_min']:
        alerts['accuracy'] = f"Accuracy below threshold: {current_performance_metrics['accuracy']:.4f}"
    if ks_stat_f0 > alert_thresholds['ks_max']:
        alerts['ks_feature_0'] = f"K-S Stat (Feature 0) above threshold: {ks_stat_f0:.4f}"
    if jsd_stat_f0 > alert_thresholds['jsd_max']:
        alerts['jsd_feature_0'] = f"JSD (Feature 0) above threshold: {jsd_stat_f0:.4f}"

    return {
        'time_step': time_step,
        'accuracy': current_performance_metrics['accuracy'],
        'precision': current_performance_metrics['precision'],
        'recall': current_performance_metrics['recall'],
        'f1_score': current_performance_metrics['f1_score'],
        'ks_statistic_feature_0': ks_stat_f0,
        'jsd_statistic_feature_0': jsd_stat_f0,
        'alerts': alerts,
        'monitoring_data_X': new_monitoring_data_X_df,
        'monitoring_data_y': new_monitoring_data_y_arr,
        'drift_params_applied': drift_params.copy()
    }

def retrain_challenger_model(X_train_challenger, y_train_challenger, random_state=42):
    """Trains a new Challenger model."""
    model = LogisticRegression(random_state=random_state, solver='liblinear')
    model.fit(X_train_challenger, y_train_challenger)
    return model

def compare_champion_challenger(champion_metrics, challenger_metrics, champion_drift_metrics, challenger_drift_metrics):
    """
    Generates plotly bar charts for model performance and drift comparison.
    Expects champion_drift_metrics and challenger_drift_metrics to contain
    performance and drift metrics for consistent plotting.
    """
    # Performance Metrics Comparison
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
    perf_data = []
    for metric in metrics_to_compare:
        perf_data.append({'Metric': metric, 'Model': 'Champion', 'Score': champion_metrics.get(metric, 0.0)})
        perf_data.append({'Metric': metric, 'Model': 'Challenger', 'Score': challenger_metrics.get(metric, 0.0)})
    perf_df = pd.DataFrame(perf_data)

    fig_perf_comp = go.Figure()
    for i, model_name in enumerate(['Champion', 'Challenger']):
        subset = perf_df[perf_df['Model'] == model_name]
        fig_perf_comp.add_trace(go.Bar(
            x=subset['Metric'],
            y=subset['Score'],
            name=model_name,
            marker_color= ('blue' if model_name == 'Champion' else 'red'),
            offsetgroup=i
        ))
    fig_perf_comp.update_layout(
        barmode='group',
        title_text='Champion vs. Challenger Performance Comparison',
        yaxis_title='Score',
        xaxis_title='Metric Type'
    )

    # Drift Metrics Comparison
    drift_metrics_to_compare = ['ks_statistic_feature_0', 'jsd_statistic_feature_0']
    drift_data = []
    for metric_key in drift_metrics_to_compare:
        # Ensure we use the correct keys from the full dicts passed
        drift_data.append({'Metric': metric_key.replace("_", " ").title(), 'Model': 'Champion', 'Statistic': champion_drift_metrics.get(metric_key, 0.0)})
        drift_data.append({'Metric': metric_key.replace("_", " ").title(), 'Model': 'Challenger', 'Statistic': challenger_drift_metrics.get(metric_key, 0.0)})
    drift_df = pd.DataFrame(drift_data)

    fig_drift_comp = go.Figure()
    for i, model_name in enumerate(['Champion', 'Challenger']):
        subset = drift_df[drift_df['Model'] == model_name]
        fig_drift_comp.add_trace(go.Bar(
            x=subset['Metric'],
            y=subset['Statistic'],
            name=model_name,
            marker_color=('blue' if model_name == 'Champion' else 'red'),
            offsetgroup=i
        ))
    fig_drift_comp.update_layout(
        barmode='group',
        title_text='Champion vs. Challenger Data Drift Metrics Comparison (Feature 0)',
        yaxis_title='Drift Statistic',
        xaxis_title='Drift Metric'
    )

    return fig_perf_comp, fig_drift_comp

def log_manual_review(time_step, alert_details):
    """Records a manual review event."""
    log_entry = f"Time Step {time_step}: Manual review triggered. Alert details: {alert_details}"
    if 'manual_review_log' not in st.session_state:
        st.session_state.manual_review_log = []
    st.session_state.manual_review_log.append(log_entry)
    return log_entry
