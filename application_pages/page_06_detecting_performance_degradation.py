
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import run_monitoring_step

def main():
    st.markdown(
        """
        ## Detecting Performance Degradation

        As a Risk Manager, you know that models, like any other asset, can degrade over time. This page simulates a scenario where our Champion model's performance begins to decline in production. This degradation can be due to various reasons, such as shifts in customer behavior, changes in external economic factors, or even subtle issues within data pipelines.

        ### Persona's Action: Identifying a Performance Issue
        Your objective is to quickly identify when the model's performance falls below acceptable thresholds. This early detection allows for timely intervention, preventing significant financial or operational impact. You will observe how the performance metrics, particularly accuracy, start to trend downwards, signaling a critical change in the model's effectiveness.
        """
    )

    # Continue simulation with performance degradation for steps 6-15
    if st.session_state.max_time_step_reached < 15:
        st.info(f"Simulating performance degradation for the next {15 - st.session_state.max_time_step_reached} monitoring steps...")
        for t_step in range(max(6, st.session_state.max_time_step_reached + 1), 16):
            current_drift_params = st.session_state.drift_parameters_stable.copy()
            current_drift_params['performance_degradation_factor'] = min(0.01 * (t_step - 5), 0.15) # Gradually degrade performance
            
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
        st.success(f"Completed monitoring steps up to {st.session_state.max_time_step_reached}, observing performance degradation.")
    else:
        st.info(f"Currently displaying monitoring data up to time step {st.session_state.max_time_step_reached}, including performance degradation.")

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
        <i>The plot above shows the time-series evolution of key classification performance metrics. Observe how the Accuracy (blue line) begins to trend downwards, indicating degradation.</i>
        </p>
    """, unsafe_allow_html=True)

    # Plotting Drift Metrics (these should remain stable in this degradation step)
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
        <i>In this scenario, while performance degrades, the raw data distributions (monitored by K-S and JSD) remain relatively stable, suggesting the degradation might stem from factors other than direct input data shift.</i>
        </p>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        Performance degradation is a clear signal that something is amiss. As a Risk Manager, your next step is to understand if this degradation is accompanied by shifts in the input data itself, which is often a root cause.
        """
    )

    if st.button("Investigate Data Drift"):
        st.session_state.current_page = "Detecting Data Drift"
        st.rerun()
