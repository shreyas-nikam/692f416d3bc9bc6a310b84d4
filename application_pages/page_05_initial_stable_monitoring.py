
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import run_monitoring_step

def main():
    st.markdown(
        """
        ## Initial Stable Monitoring

        As a Risk Manager, your commitment to continuous validation begins now. The Champion model, having been trained and its baseline performance documented, is now in production. Your immediate task is to observe its behavior under normal operating conditions to confirm its stability before any significant external factors come into play.

        ### Persona's Action: Verifying Operational Stability
        In this phase, you are looking for consistency. The model should maintain its performance, and the incoming data should closely resemble the baseline. This initial period of stable monitoring builds confidence in the model's robustness and helps validate your baseline assumptions. It's about establishing a "normal" operational rhythm for your monitoring dashboard.
        """
    )

    # Initialize monitoring state if not present
    if 'historical_metrics' not in st.session_state:
        st.session_state.historical_metrics = []
        st.session_state.monitoring_data_X = pd.DataFrame(columns=[f'feature_{i}' for i in range(st.session_state.baseline_X.shape[1])])
        st.session_state.monitoring_data_y = np.array([])
        st.session_state.current_alert_thresholds = {'accuracy_min': 0.85, 'ks_max': 0.15, 'jsd_max': 0.15}
        st.session_state.drift_parameters_stable = {'mean_shift': 0, 'std_factor': 1, 'concept_drift_factor': 0, 'performance_degradation_factor': 0}
        st.session_state.max_time_step_reached = 0
        st.session_state.simulation_logs = {'alerts': []} # Initialize for alerts log
        st.session_state.manual_review_log = [] # Initialize manual review log
        st.session_state.compare_button_enabled = False
        st.session_state.promote_button_enabled = False

    # Simulate 5 stable steps if not already done
    if st.session_state.max_time_step_reached < 5:
        st.info(f"Simulating initial {5 - st.session_state.max_time_step_reached} stable monitoring steps...")
        for t_step in range(st.session_state.max_time_step_reached + 1, 6):
            step_results = run_monitoring_step(
                st.session_state.champion_model,
                st.session_state.monitoring_data_X,
                st.session_state.monitoring_data_y,
                st.session_state.baseline_X,
                st.session_state.baseline_y,
                t_step,
                st.session_state.drift_parameters_stable, # Stable drift params
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
        st.success(f"Completed {st.session_state.max_time_step_reached} stable monitoring steps.")
    else:
        st.info(f"Currently displaying monitoring data up to time step {st.session_state.max_time_step_reached}.")

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

    # Add accuracy threshold line
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
        <i>The plot above shows the time-series evolution of key classification performance metrics (Accuracy, Precision, Recall, F1-Score). A horizontal red dashed line indicates the minimum acceptable accuracy threshold.</i>
        </p>
    """, unsafe_allow_html=True)


    # Plotting Drift Metrics
    fig_drift = go.Figure()
    fig_drift.add_trace(go.Scatter(x=time_steps, y=ks_stats, mode='lines+markers', name='K-S Statistic (Feature 0)',
                                   hovertemplate='Time Step: %{x}<br>K-S Stat: %{y:.4f}<extra></extra>'))
    fig_drift.add_trace(go.Scatter(x=time_steps, y=jsd_stats, mode='lines+markers', name='JSD (Feature 0)',
                                   hovertemplate='Time Step: %{x}<br>JSD: %{y:.4f}<extra></extra>'))

    # Add K-S and JSD thresholds
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
        <i>The plot above tracks data drift for 'Feature 0' using the Kolmogorov-Smirnov (K-S) statistic and Jensen-Shannon Divergence (JSD). Horizontal dashed lines indicate the maximum tolerable thresholds for these metrics.</i>
        </p>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        These initial monitoring steps demonstrate the model operating under stable conditions. Both performance and data drift metrics remain within acceptable bounds, reaffirming the model's health post-deployment. However, the real world is dynamic. Your vigilance as a Risk Manager is paramount as conditions change.
        """
    )

    if st.button("Simulate Performance Degradation"):
        st.session_state.current_page = "Detecting Performance Degradation"
        st.rerun()
