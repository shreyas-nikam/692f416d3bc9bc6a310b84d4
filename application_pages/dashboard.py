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
        flip_indices = np.random.choice(np.where(drift_effect == 1)[0], size=int(
            num_samples * concept_drift_factor), replace=False)
        y[flip_indices] = 1 - y[flip_indices]

    if performance_degradation_factor > 0:
        num_mislabels = int(num_samples * performance_degradation_factor)
        mislab_indices = np.random.choice(
            num_samples, size=num_mislabels, replace=False)
        y[mislab_indices] = 1 - y[mislab_indices]

    return X, y


@st.cache_resource
def train_logistic_regression_model(X_train, y_train, random_state=42):
    # Added max_iter for convergence
    model = LogisticRegression(
        random_state=random_state, solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)
    return model


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
    baseline_feature_data = baseline_feature_data[~np.isnan(
        baseline_feature_data)]
    current_feature_data = current_feature_data[~np.isnan(
        current_feature_data)]
    if len(baseline_feature_data) == 0 or len(current_feature_data) == 0:
        return 1.0, 1.0  # Max KS statistic if one distribution is empty
    ks_statistic, p_value = ks_2samp(
        baseline_feature_data, current_feature_data)
    return ks_statistic, p_value


@st.cache_data(ttl="2h")
def calculate_data_drift_jsd(baseline_feature_data, current_feature_data, num_bins=50):
    baseline_feature_data = np.asarray(baseline_feature_data)
    current_feature_data = np.asarray(current_feature_data)
    baseline_feature_data = baseline_feature_data[~np.isnan(
        baseline_feature_data)]
    current_feature_data = current_feature_data[~np.isnan(
        current_feature_data)]
    if len(baseline_feature_data) == 0 or len(current_feature_data) == 0:
        return 1.0  # Max JSD if one distribution is empty

    all_data = np.concatenate([baseline_feature_data, current_feature_data])
    min_val, max_val = np.min(all_data), np.max(all_data)
    # Ensure min_val and max_val are not the same if all_data has only one unique value
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5
    bins = np.linspace(min_val, max_val, num_bins + 1)

    hist_baseline, _ = np.histogram(
        baseline_feature_data, bins=bins, density=True)
    hist_current, _ = np.histogram(
        current_feature_data, bins=bins, density=True)

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
        performance_degradation_factor=drift_params.get(
            'performance_degradation_factor', 0)
    )
    current_batch_df = pd.DataFrame(current_batch_X, columns=[
                                    f'feature_{i}' for i in range(baseline_X.shape[1])])

    if historical_data_X.empty:
        monitoring_data_X = current_batch_df
        monitoring_data_y = current_batch_y
    else:
        monitoring_data_X = pd.concat(
            [historical_data_X, current_batch_df], ignore_index=True)
        monitoring_data_y = np.concatenate(
            [historical_data_y, current_batch_y])

    y_pred_current = predict_with_model(champion_model, current_batch_X)
    performance_metrics = calculate_classification_metrics(
        current_batch_y, y_pred_current)

    ks_statistic_f0, _ = calculate_data_drift_ks(
        baseline_X[:, 0], current_batch_X[:, 0])
    jsd_statistic_f0 = calculate_data_drift_jsd(
        baseline_X[:, 0], current_batch_X[:, 0])

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
        results['alerts'][
            'accuracy'] = f"Accuracy threshold crossed: {performance_metrics['accuracy']:.4f} (Threshold: {alert_thresholds['accuracy_min']:.4f})"
    if ks_statistic_f0 > alert_thresholds['ks_max']:
        results['alerts'][
            'ks_feature_0'] = f"K-S Stat (Feature 0) threshold crossed: {ks_statistic_f0:.4f} (Threshold: {alert_thresholds['ks_max']:.4f})"
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
        go.Bar(name='Champion', x=metrics_df.columns, y=metrics_df.loc['Champion'], marker_color='blue',
               hovertemplate='Model: Champion<br>Metric: %{x}<br>Value: %{y:.4f}<extra></extra>'),
        go.Bar(name='Challenger', x=metrics_df.columns, y=metrics_df.loc['Challenger'], marker_color='red',
               hovertemplate='Model: Challenger<br>Metric: %{x}<br>Value: %{y:.4f}<extra></extra>')
    ])
    fig_perf_comp.update_layout(
        barmode='group', title='Performance Metrics: Champion vs. Challenger', yaxis_title='Score')
    st.plotly_chart(fig_perf_comp)

    drift_df = pd.DataFrame({
        'Champion': champion_drift_metrics,
        'Challenger': challenger_drift_metrics
    }).T
    drift_df.index.name = 'Model'

    fig_drift_comp = go.Figure(data=[
        go.Bar(name='Champion', x=drift_df.columns, y=drift_df.loc['Champion'], marker_color='blue',
               hovertemplate='Model: Champion<br>Metric: %{x}<br>Value: %{y:.4f}<extra></extra>'),
        go.Bar(name='Challenger', x=drift_df.columns, y=drift_df.loc['Challenger'], marker_color='red',
               hovertemplate='Model: Challenger<br>Metric: %{x}<br>Value: %{y:.4f}<extra></extra>')
    ])
    fig_drift_comp.update_layout(
        barmode='group', title='Drift Metrics: Champion vs. Challenger (on Recent Data)', yaxis_title='Drift Statistic')
    st.plotly_chart(fig_drift_comp)


def log_manual_review(time_step, alert_details):
    st.info(
        f"Manual Review Log: Time Step {time_step}, Alert Details: {alert_details}")


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
            'accuracy_min': 0.70,  # Default based on typical scenarios
            'ks_max': 0.30,      # Default based on typical scenarios
            'jsd_max': 0.20      # Default based on typical scenarios
        }
        st.session_state.initial_champion_trained = False
        st.session_state.random_state = 42  # Default random state


def main():
    init_session_state()

    st.header("AI Model Health Dashboard Simulator")

    st.markdown(r"""
    This interactive Streamlit dashboard provides a simulated environment for monitoring AI model health,
    designed specifically for **Risk Managers**. It aims to demonstrate key principles of AI Model Risk Management (MRM),
    aligning with SR 11-7 guidelines for adaptive AI systems.

    **Learning Goals:**
    *   **Continuous Monitoring**: Implement a system to track model performance and data characteristics over time.
    *   **Drift Detection**: Identify and distinguish between performance degradation, data drift ($P(X)$ change), and concept drift ($P(Y|X)$ change).
    *   **Threshold-based Alerts**: Configure and react to alerts when model health metrics deviate from acceptable norms.
    *   **Intervention Strategies**: Simulate model retraining, Champion-Challenger comparison, and manual review processes.
    *   **SR 11-7 Principles**: Reinforce ongoing validation and risk control for adaptive AI systems through practical simulation.
    """)
    st.divider()

    st.subheader("1. Baseline Setup")
    st.markdown(r"""
    In this section, we set up our initial environment by generating synthetic baseline data and training our first "Champion" model.
    This baseline will serve as the reference point for all subsequent monitoring and drift detection.
    """)

    with st.expander("Generate Baseline Data & Train Champion Model"):
        st.markdown("Configure parameters for synthetic data generation:")
        col1, col2, col3 = st.columns(3)
        with col1:
            num_samples = st.number_input(
                "Number of Baseline Samples", min_value=100, value=1000, step=100)
            num_features = st.number_input(
                "Number of Features", min_value=2, value=10, step=1)
        with col2:
            n_informative = st.number_input(
                "Number of Informative Features", min_value=1, value=3, step=1)
            n_redundant = st.number_input(
                "Number of Redundant Features", min_value=0, value=0, step=1)
        with col3:
            n_clusters_per_class = st.number_input(
                "Clusters per Class", min_value=1, value=1, step=1)
            random_state = st.number_input(
                "Random State", min_value=0, value=42, step=1)

        if st.button("Generate Baseline Data and Train Initial Champion Model"):
            st.session_state.baseline_X, st.session_state.baseline_y = generate_synthetic_classification_data(
                num_samples=num_samples, num_features=num_features, n_informative=n_informative,
                n_redundant=n_redundant, n_clusters_per_class=n_clusters_per_class, random_state=random_state
            )
            st.session_state.baseline_df = pd.DataFrame(st.session_state.baseline_X, columns=[
                                                        f'feature_{i}' for i in range(num_features)])
            st.session_state.baseline_df['target'] = st.session_state.baseline_y
            st.session_state.random_state = random_state  # Store random state
            st.success("Baseline data generated successfully!")

            # Train initial Champion model
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state.baseline_X, st.session_state.baseline_y, test_size=0.2, random_state=random_state)
            st.session_state.champion_model = train_logistic_regression_model(
                X_train, y_train, random_state)
            y_pred_baseline = predict_with_model(
                st.session_state.champion_model, X_test)
            st.session_state.baseline_metrics = calculate_classification_metrics(
                y_test, y_pred_baseline)
            st.session_state.initial_champion_trained = True
            st.success("Initial Champion Model trained successfully!")

            # Reset monitoring data and history
            st.session_state.historical_accuracy = [
                st.session_state.baseline_metrics['accuracy']]
            st.session_state.historical_precision = [
                st.session_state.baseline_metrics['precision']]
            st.session_state.historical_recall = [
                st.session_state.baseline_metrics['recall']]
            st.session_state.historical_f1 = [
                st.session_state.baseline_metrics['f1_score']]
            st.session_state.historical_ks_stats_f0 = [
                0.0]  # K-S stat is 0 at baseline
            st.session_state.historical_jsd_stats_f0 = [
                0.0]  # JSD stat is 0 at baseline
            st.session_state.historical_alerts = []
            st.session_state.monitoring_data_X = pd.DataFrame()
            st.session_state.monitoring_data_y = np.array([])
            st.session_state.current_t_step = 0
            st.session_state.challenger_model = None  # Reset Challenger as well

    if st.session_state.baseline_X is not None and st.session_state.initial_champion_trained:
        st.subheader("Baseline Data Snapshot:")
        st.write(st.session_state.baseline_df.head())

        st.markdown(r"""
        The initial Champion model is trained on the baseline data. Its performance metrics on a holdout set are:
        """)
        st.write(pd.DataFrame([st.session_state.baseline_metrics]).round(4))

        st.markdown(r"""
        **Baseline Target Distribution:**
        """)
        target_distribution = st.session_state.baseline_df['target'].value_counts(
            normalize=True).reset_index()
        target_distribution.columns = ['Target Class', 'Proportion']
        st.dataframe(target_distribution.round(4))
    else:
        st.info(
            "Please generate baseline data and train the initial Champion model to proceed.")

    st.divider()

    st.subheader("2. Continuous Monitoring & Alerts")
    st.markdown(r"""
    In this section, we simulate a continuous production environment where new data batches arrive over time.
    The dashboard monitors the Champion model's performance and detects data or concept drift.

    **Model Performance Over Time:**
    The following metrics are tracked: Accuracy, Precision, Recall, and F1-Score.
    *   **Accuracy:** $A = \frac{TP + TN}{TP + TN + FP + FN}$
    *   **Precision:** $P = \frac{TP}{TP + FP}$
    *   **Recall:** $R = \frac{TP}{TP + FN}$
    *   **F1-Score:** $F1 = 2 \times \frac{P \times R}{P + R}$
    """)

    with st.expander("Configure Simulation Parameters"):
        st.markdown(
            "Adjust the parameters below to introduce different types of drift or degradation into the simulated data stream.")
        st.session_state.drift_parameters['mean_shift'] = st.slider(
            "Mean Shift (Feature 0 - Data Drift)", min_value=-2.0, max_value=2.0, value=st.session_state.drift_parameters['mean_shift'], step=0.1)
        st.session_state.drift_parameters['std_factor'] = st.slider(
            "Std Dev Factor (Feature 0 - Data Drift)", min_value=0.5, max_value=2.0, value=st.session_state.drift_parameters['std_factor'], step=0.1)
        st.session_state.drift_parameters['concept_drift_factor'] = st.slider("Concept Drift Factor", min_value=0.0, max_value=0.5, value=st.session_state.drift_parameters['concept_drift_factor'], step=0.01,
                                                                              help="Proportion of labels to flip based on feature 0. Simulates P(Y|X) change.")
        st.session_state.drift_parameters['performance_degradation_factor'] = st.slider("Performance Degradation Factor", min_value=0.0, max_value=0.5, value=st.session_state.drift_parameters['performance_degradation_factor'], step=0.01,
                                                                                        help="Proportion of random labels to flip. Simulates overall performance degradation independent of input features.")

    if st.session_state.champion_model:
        num_steps_to_simulate = st.number_input(
            "Number of steps to simulate (per click)", min_value=1, value=10, step=1)
        if st.button("Run Simulation Steps"):
            for _ in range(num_steps_to_simulate):
                st.session_state.current_t_step += 1
                monitoring_results = run_monitoring_step(
                    st.session_state.champion_model,
                    st.session_state.monitoring_data_X,
                    st.session_state.monitoring_data_y,
                    st.session_state.baseline_X,
                    st.session_state.baseline_y,
                    st.session_state.current_t_step,
                    st.session_state.drift_parameters,
                    st.session_state.alert_thresholds
                )
                st.session_state.historical_accuracy.append(
                    monitoring_results['accuracy'])
                st.session_state.historical_precision.append(
                    monitoring_results['precision'])
                st.session_state.historical_recall.append(
                    monitoring_results['recall'])
                st.session_state.historical_f1.append(
                    monitoring_results['f1_score'])
                st.session_state.historical_ks_stats_f0.append(
                    monitoring_results['ks_statistic_feature_0'])
                st.session_state.historical_jsd_stats_f0.append(
                    monitoring_results['jsd_statistic_feature_0'])
                st.session_state.monitoring_data_X = monitoring_results['monitoring_data_X']
                st.session_state.monitoring_data_y = monitoring_results['monitoring_data_y']
                if monitoring_results['alerts']:
                    st.session_state.historical_alerts.append({
                        'time_step': st.session_state.current_t_step,
                        'alerts': monitoring_results['alerts']
                    })
            st.success(
                f"Simulated {num_steps_to_simulate} steps. Current Time Step: {st.session_state.current_t_step}")

        # Plot Model Performance Over Time
        if st.session_state.historical_accuracy:
            fig_perf = go.Figure()
            time_steps = list(range(len(st.session_state.historical_accuracy)))

            fig_perf.add_trace(go.Scatter(x=time_steps, y=st.session_state.historical_accuracy, mode='lines+markers', name='Accuracy',
                                          hovertemplate='Step: %{x}<br>Accuracy: %{y:.4f}<extra></extra>'))
            fig_perf.add_trace(go.Scatter(x=time_steps, y=st.session_state.historical_precision, mode='lines+markers', name='Precision',
                                          hovertemplate='Step: %{x}<br>Precision: %{y:.4f}<extra></extra>'))
            fig_perf.add_trace(go.Scatter(x=time_steps, y=st.session_state.historical_recall, mode='lines+markers', name='Recall',
                                          hovertemplate='Step: %{x}<br>Recall: %{y:.4f}<extra></extra>'))
            fig_perf.add_trace(go.Scatter(x=time_steps, y=st.session_state.historical_f1, mode='lines+markers', name='F1-Score',
                                          hovertemplate='Step: %{x}<br>F1-Score: %{y:.4f}<extra></extra>'))

            fig_perf.add_hline(y=st.session_state.alert_thresholds['accuracy_min'], line_dash="dot",
                               annotation_text=f"Accuracy Threshold: {st.session_state.alert_thresholds['accuracy_min']:.2f}",
                               annotation_position="bottom right",
                               line_color="red")

            fig_perf.update_layout(title='Model Performance Over Time',
                                   xaxis_title='Time Step',
                                   yaxis_title='Metric Value',
                                   legend_title='Metric')
            st.plotly_chart(fig_perf)

        st.markdown(r"""
        **Understanding and Detecting Data Drift (Covariate Shift):**
        Data drift occurs when the statistical properties of the input features $P(X)$ change over time.
        We use the following metrics to detect data drift:

        *   **Kolmogorov-Smirnov (K-S) Statistic:** Measures the maximum vertical distance between the empirical cumulative distribution functions (CDFs) of two samples.
            $$D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)|$$
            Where $F_{1,n}(x)$ and $F_{2,m}(x)$ are the empirical CDFs of the two samples.

        *   **Jensen-Shannon Divergence (JSD):** A symmetric and smoothed version of the Kullback-Leibler divergence, measuring the similarity between two probability distributions.
            $$JSD(P||Q) = \frac{1}{2} D_{KL}(P||M) + \frac{1}{2} D_{KL}(Q||M)$$
            Where $M = \frac{1}{2}(P+Q)$ and $D_{KL}(P||Q) = \sum_{i} P(i) \log \left( \frac{P(i)}{Q(i)} \right)$.
        """)

        # Plot Drift Metrics Over Time
        if st.session_state.historical_ks_stats_f0 and st.session_state.historical_jsd_stats_f0:
            fig_drift = go.Figure()
            time_steps = list(
                range(len(st.session_state.historical_ks_stats_f0)))

            fig_drift.add_trace(go.Scatter(x=time_steps, y=st.session_state.historical_ks_stats_f0, mode='lines+markers', name='K-S Statistic (Feature 0)',
                                           hovertemplate='Step: %{x}<br>K-S Stat: %{y:.4f}<extra></extra>', yaxis='y1'))
            fig_drift.add_hline(y=st.session_state.alert_thresholds['ks_max'], line_dash="dot",
                                annotation_text=f"K-S Threshold: {st.session_state.alert_thresholds['ks_max']:.2f}",
                                annotation_position="bottom left",
                                line_color="orange", yref='y1')

            fig_drift.add_trace(go.Scatter(x=time_steps, y=st.session_state.historical_jsd_stats_f0, mode='lines+markers', name='JSD (Feature 0)',
                                           hovertemplate='Step: %{x}<br>JSD: %{y:.4f}<extra></extra>', yaxis='y2'))
            fig_drift.add_hline(y=st.session_state.alert_thresholds['jsd_max'], line_dash="dot",
                                annotation_text=f"JSD Threshold: {st.session_state.alert_thresholds['jsd_max']:.2f}",
                                annotation_position="top left",
                                line_color="purple", yref='y2')

            fig_drift.update_layout(title='Drift Metrics Over Time (Feature 0)',
                                    xaxis_title='Time Step',
                                    yaxis=dict(title='K-S Statistic'),
                                    yaxis2=dict(
                                        title='JSD', overlaying='y', side='right'),
                                    legend_title='Metric')
            st.plotly_chart(fig_drift)

        st.markdown(r"""
        **Configuring Alert Thresholds and Triggering Alerts:**
        Alerts are triggered when monitored metrics (Accuracy, K-S Statistic, JSD) cross predefined thresholds.
        Adjust the sliders below to set these thresholds.
        """)

        with st.expander("Configure Alert Thresholds"):
            st.session_state.alert_thresholds['accuracy_min'] = st.slider(
                'Min. Accuracy Threshold', min_value=0.0, max_value=1.0, value=st.session_state.alert_thresholds['accuracy_min'], step=0.01)
            st.session_state.alert_thresholds['ks_max'] = st.slider(
                'Max. K-S Stat Threshold', min_value=0.0, max_value=1.0, value=st.session_state.alert_thresholds['ks_max'], step=0.01)
            st.session_state.alert_thresholds['jsd_max'] = st.slider(
                'Max. JSD Threshold', min_value=0.0, max_value=1.0, value=st.session_state.alert_thresholds['jsd_max'], step=0.01)

        st.subheader("Active Alerts:")
        if st.session_state.historical_alerts:
            for alert_entry in st.session_state.historical_alerts:
                st.warning(f"Time Step {alert_entry['time_step']}:")
                for alert_type, alert_message in alert_entry['alerts'].items():
                    st.write(f"- {alert_message}")
        else:
            st.info("No active alerts at this time.")

        st.markdown(r"""
        **Feature Distribution: Baseline vs. Current (Feature 0)**
        This histogram visually compares the distribution of `feature_0` from the initial baseline data
        against the accumulated `monitoring_data_X` from the simulated production batches.
        This helps in understanding the nature and extent of data drift for this specific feature.
        """)

        if not st.session_state.monitoring_data_X.empty:
            fig_hist = go.Figure()
            # Ensure baseline_X is not None before accessing it
            if st.session_state.baseline_X is not None:
                fig_hist.add_trace(go.Histogram(
                    x=st.session_state.baseline_X[:, 0], name='Baseline Feature 0', opacity=0.7, histnorm='probability density'))
            fig_hist.add_trace(go.Histogram(
                x=st.session_state.monitoring_data_X['feature_0'], name='Current Feature 0', opacity=0.7, histnorm='probability density'))
            fig_hist.update_layout(barmode='overlay', title='Feature 0 Distribution: Baseline vs. Current',
                                   xaxis_title='Feature 0 Value', yaxis_title='Probability Density')
            st.plotly_chart(fig_hist)
        else:
            st.info(
                "Run some simulation steps to visualize current feature distributions.")

    else:
        st.info(
            "Please generate baseline data and train the initial Champion model to start monitoring.")

    st.divider()

    st.subheader("3. Intervention Strategies")
    st.markdown(r"""
    When alerts are triggered or model health deteriorates, intervention strategies are crucial.
    This section simulates retraining a challenger model, comparing it against the champion, and promoting it if better.

    **Simulating Retraining a "Challenger" Model:**
    This simulates training a new model on the *most recent* data (or a combination of baseline and recent data)
    to potentially mitigate drift or degradation.
    """)

    if st.session_state.champion_model:
        if st.button("Simulate Retraining Challenger Model"):
            if not st.session_state.monitoring_data_X.empty:
                # For simplicity, let's retrain on all accumulated monitoring data
                X_retrain = st.session_state.monitoring_data_X.values
                y_retrain = st.session_state.monitoring_data_y
                # Use st.session_state.random_state for consistency or derive a new one
                challenger_random_state = st.session_state.random_state + \
                    1 if 'random_state' in st.session_state else 43
                st.session_state.challenger_model = train_logistic_regression_model(
                    X_retrain, y_retrain, random_state=challenger_random_state)
                st.success(
                    "Challenger Model retrained successfully on recent monitoring data!")
            else:
                st.warning(
                    "No monitoring data accumulated yet. Run some simulation steps first.")
    else:
        st.info(
            "Please generate baseline data and train the initial Champion model first.")

    st.markdown(r"""
    **Champion-Challenger Comparison and Model Promotion:**
    After retraining, the Challenger model is compared against the current Champion model using recent data
    to determine if it offers improved performance or better handles the observed drift.
    """)

    if st.session_state.champion_model and st.session_state.challenger_model and not st.session_state.monitoring_data_X.empty:
        if st.button("Compare Champion vs. Challenger"):
            st.info(
                "Comparing Champion and Challenger models on the latest monitoring data batch...")

            # Evaluate Champion
            y_pred_champion = predict_with_model(
                st.session_state.champion_model, st.session_state.monitoring_data_X.values)
            champion_metrics_current = calculate_classification_metrics(
                st.session_state.monitoring_data_y, y_pred_champion)
            champion_ks_f0, _ = calculate_data_drift_ks(
                st.session_state.baseline_X[:, 0], st.session_state.monitoring_data_X['feature_0'])
            champion_jsd_f0 = calculate_data_drift_jsd(
                st.session_state.baseline_X[:, 0], st.session_state.monitoring_data_X['feature_0'])
            champion_drift_metrics_current = {
                'K-S Stat (F0)': champion_ks_f0, 'JSD (F0)': champion_jsd_f0}

            # Evaluate Challenger
            y_pred_challenger = predict_with_model(
                st.session_state.challenger_model, st.session_state.monitoring_data_X.values)
            challenger_metrics_current = calculate_classification_metrics(
                st.session_state.monitoring_data_y, y_pred_challenger)
            challenger_ks_f0, _ = calculate_data_drift_ks(
                st.session_state.baseline_X[:, 0], st.session_state.monitoring_data_X['feature_0'])
            challenger_jsd_f0 = calculate_data_drift_jsd(
                st.session_state.baseline_X[:, 0], st.session_state.monitoring_data_X['feature_0'])
            challenger_drift_metrics_current = {
                'K-S Stat (F0)': challenger_ks_f0, 'JSD (F0)': challenger_jsd_f0}

            compare_champion_challenger(champion_metrics_current, challenger_metrics_current,
                                        champion_drift_metrics_current, challenger_drift_metrics_current)

            if challenger_metrics_current['accuracy'] > champion_metrics_current['accuracy']:
                st.success(
                    "Challenger model shows better accuracy! Consider promoting it.")
            else:
                st.info("Champion model maintains better or similar accuracy.")

        if st.button("Promote Challenger to Champion"):
            # Ensure the comparison was just run or the user is aware of the current state
            # For simplicity, we assume metrics are already evaluated if button is pressed
            if st.session_state.challenger_model is not None:
                st.session_state.champion_model = st.session_state.challenger_model
                # Reset challenger after promotion, it needs to be retrained for next cycle
                st.session_state.challenger_model = None
                # Optionally, re-evaluate baseline metrics with the new champion on the *current* monitoring data to set a new reference for future comparison
                # Or, clear history and restart monitoring to establish a new 'stable' baseline performance.
                # For this simulation, clearing history and resetting step count makes the next phase clearer.
                y_pred_new_champion_on_current_data = predict_with_model(
                    st.session_state.champion_model, st.session_state.monitoring_data_X.values)
                st.session_state.baseline_metrics = calculate_classification_metrics(
                    st.session_state.monitoring_data_y, y_pred_new_champion_on_current_data)

                st.session_state.historical_accuracy = [
                    st.session_state.baseline_metrics['accuracy']]
                st.session_state.historical_precision = [
                    st.session_state.baseline_metrics['precision']]
                st.session_state.historical_recall = [
                    st.session_state.baseline_metrics['recall']]
                st.session_state.historical_f1 = [
                    st.session_state.baseline_metrics['f1_score']]
                st.session_state.historical_ks_stats_f0 = [0.0]
                st.session_state.historical_jsd_stats_f0 = [0.0]
                st.session_state.historical_alerts = []
                st.session_state.monitoring_data_X = pd.DataFrame()
                st.session_state.monitoring_data_y = np.array([])
                st.session_state.current_t_step = 0
                st.success(
                    "Challenger model promoted to Champion! Monitoring restarted with new Champion.")
            else:
                st.warning(
                    "No Challenger model available to promote. Please retrain one first.")

    elif st.session_state.champion_model and not st.session_state.challenger_model:
        st.info("Train a Challenger model first to enable comparison.")
    elif not st.session_state.champion_model:
        st.info(
            "Please generate baseline data and train the initial Champion model first.")

    st.markdown(r"""
    **Human-in-the-Loop and Governance in AI Monitoring (SR 11-7):**
    SR 11-7 emphasizes human oversight. Manual reviews are essential for qualitative assessment
    and to understand complex drift scenarios that automated metrics might miss.
    """)

    if st.button("Trigger Manual Review"):
        alert_details = {}
        if st.session_state.historical_alerts:
            last_alerts = st.session_state.historical_alerts[-1]['alerts']
            for k, v in last_alerts.items():
                alert_details[k] = v
        else:
            alert_details['status'] = "No specific alerts, triggered manually."
        log_manual_review(st.session_state.current_t_step, alert_details)
        st.info(
            "Manual review triggered. A deeper investigation into model and data health is initiated.")

    st.divider()

    st.subheader("Conclusion: Continuous Monitoring for Trustworthy AI")
    st.markdown(r"""
    This simulator demonstrates the critical importance of continuous monitoring in AI systems for **Model Risk Management (MRM)**.
    By tracking performance, detecting drift, and implementing timely interventions, organizations can maintain the trustworthiness
    and reliability of their AI models in dynamic real-world environments, aligning with regulatory expectations like SR 11-7.

    Key takeaways include:
    *   Proactive detection of model degradation and data/concept drift.
    *   Setting and reacting to alert thresholds.
    *   The role of retraining and Champion-Challenger frameworks.
    *   The necessity of human oversight and structured intervention processes.
    """)
