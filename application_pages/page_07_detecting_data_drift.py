
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import run_monitoring_step, calculate_data_drift_ks, calculate_data_drift_jsd

def main():
    st.markdown(
        """
        ## Understanding and Detecting Data Drift (Covariate Shift)

        As a Risk Manager, recognizing data drift is paramount. Data drift, or covariate shift, occurs when the statistical properties of the input features change over time. This means the data that your model is seeing in production is different from the data it was trained on. This can significantly degrade model performance, even if the underlying relationship between features and the target remains the same.

        ### Persona's Action: Diagnosing Data Integrity Issues
        Your role is to identify these shifts in input data distributions. Understanding *how* the data has changed provides crucial context for observed performance degradation and informs decisions about model retraining or data pipeline adjustments. You will use statistical measures and visual comparisons to confirm the presence and magnitude of data drift.

        ---
        ### Defining and Calculating Data Drift Metrics
        We will use two key metrics to quantify data drift:

        *   **Kolmogorov-Smirnov (K-S) Statistic:** Measures the maximum difference between the cumulative distribution functions of two samples. A higher K-S statistic indicates a greater difference between the baseline and current data distributions.
            $$D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)|$$
            Where $F_{1,n}(x)$ and $F_{2,m}(x)$ are the empirical distribution functions for the two samples.

        *   **Jensen-Shannon Divergence (JSD):** A symmetric and smoothed version of Kullback-Leibler (KL) Divergence. It measures the similarity between two probability distributions. A higher JSD value indicates greater dissimilarity or divergence between the baseline and current data distributions.
            $$\text{JSD}(P||Q) = \frac{1}{2} D_{\text{KL}}(P||M) + \frac{1}{2} D_{\text{KL}}(Q||M)$$
            Where $M = \frac{1}{2}(P+Q)$, and $D_{\text{KL}}(P||Q) = \sum_i P(i) \log\left(\frac{P(i)}{Q(i)}\right)$ is the Kullback-Leibler Divergence.

        Higher values for both K-S and JSD indicate that the current data's distribution has diverged significantly from the baseline, signaling data drift.
        """
    )

    st.subheader("Example: Demonstrating K-S and JSD")
    st.markdown(
        """
        To illustrate these concepts, let's look at how K-S and JSD values change with and without data drift on a simple synthetic example.
        """
    )
    # Placeholder for demonstrating drift calculation on example data
    np.random.seed(42)
    sample_a = np.random.normal(loc=0, scale=1, size=100)
    sample_b_no_drift = np.random.normal(loc=0, scale=1, size=100)
    sample_c_with_drift = np.random.normal(loc=1, scale=1.2, size=100)
    
    ks_no_drift, p_no_drift = calculate_data_drift_ks(sample_a, sample_b_no_drift)
    jsd_no_drift = calculate_data_drift_jsd(sample_a, sample_b_no_drift)
    ks_with_drift, p_with_drift = calculate_data_drift_ks(sample_a, sample_c_with_drift)
    jsd_with_drift = calculate_data_drift_jsd(sample_a, sample_c_with_drift)
    
    st.write(f"K-S (no drift): `{ks_no_drift:.4f}`, JSD (no drift): `{jsd_no_drift:.4f}`")
    st.write(f"K-S (with drift): `{ks_with_drift:.4f}`, JSD (with drift): `{jsd_with_drift:.4f}`")
    st.markdown(
        """
        The example demonstrates that even subtle shifts in data distributions lead to higher K-S and JSD statistics, indicating potential data drift. This aligns with your goal of identifying these changes as a Risk Manager.
        """
    )

    # Simulate steps 16-30 with data drift (mean shift), reset other drifts
    if st.session_state.max_time_step_reached < 30:
        st.info(f"Simulating data drift for the next {30 - st.session_state.max_time_step_reached} monitoring steps (Time Steps {st.session_state.max_time_step_reached + 1}-{30})...")
        for t_step in range(max(16, st.session_state.max_time_step_reached + 1), 31):
            current_drift_params = st.session_state.drift_parameters_stable.copy() # Start from stable
            current_drift_params['mean_shift'] = min(0.1 * (t_step - 15), 1.0) # Gradually shift mean for feature 0
            # Ensure other drift factors are reset for this specific scenario demonstration
            current_drift_params['performance_degradation_factor'] = 0
            current_drift_params['concept_drift_factor'] = 0

            step_results = run_monitoring_step(
                st.session_state.champion_model,
                st.session_state.monitoring_data_X,
                st.session_state.monitoring_data_y,
                st.session_state.baseline_X,
                st.session_state.baseline_y,
                t_step,
                current_drift_params,
                st.session_state.current_alert_thresholds
            )
            st.session_state.historical_metrics.append({
                'time_step': t_step,
                'metrics': {k: v for k,v in step_results.items() if k not in ['alerts', 'monitoring_data_X', 'monitoring_data_y', 'drift_params_applied']},
                'alerts': step_results['alerts'],
                'drift_params_applied': step_results['drift_params_applied']
            })
            st.session_state.monitoring_data_X = step_results['monitoring_data_X']
            st.session_state.monitoring_data_y = step_results['monitoring_data_y']
            st.session_state.max_time_step_reached = t_step
            if step_results['alerts']:
                alert_message = f"ALERT at Time Step {t_step}: {step_results['alerts']}"
                st.session_state.simulation_logs['alerts'].append(alert_message)
                st.warning(alert_message)
        st.success(f"Completed monitoring steps up to {st.session_state.max_time_step_reached}, observing data drift.")
    else:
        st.info(f"Currently displaying monitoring data up to time step {st.session_state.max_time_step_reached}, including data drift.")

    # Extract data for plotting
    time_steps = [d['time_step'] for d in st.session_state.historical_metrics]
    accuracies = [d['metrics']['accuracy'] for d in st.session_state.historical_metrics]
    precisions = [d['metrics']['precision'] for d in st.session_state.historical_metrics]
    recal_scores = [d['metrics']['recall'] for d in st.session_state.historical_metrics]
    f1_scores = [d['metrics']['f1_score'] for d in st.session_state.historical_metrics]
    ks_stats = [d['metrics']['ks_statistic_feature_0'] for d in st.session_state.historical_metrics]
    jsd_stats = [d['metrics']['jsd_statistic_feature_0'] for d in st.session_state.historical_metrics]

    # Plotting Performance Metrics
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=time_steps, y=accuracies, mode='lines+markers', name='Accuracy',
                                  hovertemplate='Time Step: %{x}<br>Accuracy: %{y:.4f}<extra></extra>'))
    fig_perf.add_trace(go.Scatter(x=time_steps, y=precisions, mode='lines+markers', name='Precision',
                                  hovertemplate='Time Step: %{x}<br>Precision: %{y:.4f}<extra></extra>'))
    fig_perf.add_trace(go.Scatter(x=time_steps, y=recal_scores, mode='lines+markers', name='Recall',
                                  hovertemplate='Time Step: %{x}<br>Recall: %{y:.4f}<extra></extra>'))
    fig_perf.add_trace(go.Scatter(x=time_steps, y=f1_scores, mode='lines+markers', name='F1-Score',
                                  hovertemplate='Time Step: %{x}<br>F1-Score: %{y:.4f}<extra></extra>'))

    fig_perf.add_hline(y=st.session_state.current_alert_thresholds['accuracy_min'], line_dash="dash", line_color="red",
                       annotation_text=f"Min. Accuracy Threshold ({st.session_state.current_alert_thresholds['accuracy_min']:.2f})",
                       annotation_position="bottom right")

    fig_perf.update_layout(title='Model Performance Metrics Over Time',
                           xaxis_title='Time Step',
                           yaxis_title='Score',
                           hovermode="x unified")
    st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown(r"""
        <p style="font-size: smaller; text-align: center; color: grey;">
        <i>The plot above shows the cumulative model performance metrics. With data drift, you might observe a further decline in performance, even if direct performance degradation was reset.</i>
        </p>
    """, unsafe_allow_html=True)


    # Plotting Drift Metrics
    fig_drift = go.Figure()
    fig_drift.add_trace(go.Scatter(x=time_steps, y=ks_stats, mode='lines+markers', name='K-S Statistic (Feature 0)',
                                   hovertemplate='Time Step: %{x}<br>K-S Stat: %{y:.4f}<extra></extra>'))
    fig_drift.add_trace(go.Scatter(x=time_steps, y=jsd_stats, mode='lines+markers', name='JSD (Feature 0)',
                                   hovertemplate='Time Step: %{x}<br>JSD: %{y:.4f}<extra></extra>'))

    fig_drift.add_hline(y=st.session_state.current_alert_thresholds['ks_max'], line_dash="dash", line_color="orange",
                        annotation_text=f"Max. K-S Threshold ({st.session_state.current_alert_thresholds['ks_max']:.2f})",
                        annotation_position="bottom right")
    fig_drift.add_hline(y=st.session_state.current_alert_thresholds['jsd_max'], line_dash="dash", line_color="purple",
                        annotation_text=f"Max. JSD Threshold ({st.session_state.current_alert_thresholds['jsd_max']:.2f})",
                        annotation_position="top right")

    fig_drift.update_layout(title='Data Drift Metrics Over Time (Feature 0)',
                            xaxis_title='Time Step',
                            yaxis_title='Drift Statistic',
                            hovermode="x unified")
    st.plotly_chart(fig_drift, use_container_width=True)

    st.markdown(r"""
        <p style="font-size: smaller; text-align: center; color: grey;">
        <i>The data drift plots now clearly show an increase in both K-S Statistic and JSD for 'Feature 0', indicating that the incoming data distributions have significantly diverged from the baseline.</i>
        </p>
    """, unsafe_allow_html=True)

    st.subheader("Visualizing Feature 0 Distribution Shift")
    st.markdown(
        """
        A direct visual comparison of the feature distributions provides compelling evidence of data drift. Here, you can see the difference between the baseline distribution and the most recent batch of production data.
        """
    )

    # Overlaid Histograms for Feature 0
    baseline_feature_0 = pd.DataFrame(st.session_state.baseline_X, columns=[f'feature_{i}' for i in range(st.session_state.baseline_X.shape[1])])['feature_0']
    
    # Get the latest batch data from the monitoring_data_X for visualization
    # Assuming each step adds 100 samples (num_samples_per_step in run_monitoring_step)
    num_samples_per_step = 100 # This should match the value in run_monitoring_step
    current_monitoring_data_X_last_batch = st.session_state.monitoring_data_X.tail(num_samples_per_step)
    current_feature_0 = current_monitoring_data_X_last_batch['feature_0']

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=baseline_feature_0, name='Baseline Data', histnorm='probability density', opacity=0.7, marker_color='blue'))
    fig_hist.add_trace(go.Histogram(x=current_feature_0, name='Current Monitoring Data', histnorm='probability density', opacity=0.7, marker_color='red'))
    fig_hist.update_layout(barmode='overlay', title_text='Distribution of Feature 0: Baseline vs. Current',
                           xaxis_title='Feature 0 Value', yaxis_title='Density')
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(r"""
        <p style="font-size: smaller; text-align: center; color: grey;">
        <i>The overlaid histograms clearly show a shift in the distribution of 'Feature 0' from the baseline (blue) to the current monitoring data (red), visually confirming the data drift detected by K-S and JSD.</i>
        </p>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        Detecting data drift is a critical skill for a Risk Manager. It indicates that the model is now operating on data fundamentally different from its training environment, often leading to reduced trustworthiness and accuracy. The next challenge is to understand if the *relationship* between features and the target is changing, which is known as concept drift.
        """
    )

    if st.button("Explore Concept Drift"):
        st.session_state.current_page = "Detecting Concept Drift"
        st.rerun()
