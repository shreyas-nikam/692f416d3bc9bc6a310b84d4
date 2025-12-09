
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import run_monitoring_step

def main():
    st.markdown(
        """
        ## Understanding and Detecting Concept Drift

        As a Risk Manager, you understand that data drift is one challenge, but a more insidious one is **concept drift**. This occurs when the underlying relationship between the input features and the target variable changes over time. Unlike data drift, where the input distribution changes but the output relationship remains stable, concept drift means the very "concept" the model learned is no longer valid.

        ### Persona's Action: Uncovering Model Irrelevance
        Your objective here is to identify if the model's fundamental understanding of the world has become outdated. This often manifests as a decline in model performance even when input data distributions appear stable or when the degradation is more severe than data drift alone would explain. Detecting concept drift requires vigilance and careful analysis of performance trends.
        """
    )

    # Simulate steps 31-45 with concept drift, resetting other drifts for a focused demonstration
    if st.session_state.max_time_step_reached < 45:
        st.info(f"Simulating concept drift for the next {45 - st.session_state.max_time_step_reached} monitoring steps (Time Steps {st.session_state.max_time_step_reached + 1}-{45})...")
        for t_step in range(max(31, st.session_state.max_time_step_reached + 1), 46):
            current_drift_params = st.session_state.drift_parameters_stable.copy() # Start from stable base
            current_drift_params['concept_drift_factor'] = min(0.01 * (t_step - 30), 0.1) # Gradually introduce concept drift
            # Ensure other drift factors are reset for this specific scenario demonstration
            current_drift_params['mean_shift'] = 0
            current_drift_params['performance_degradation_factor'] = 0

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
        st.success(f"Completed monitoring steps up to {st.session_state.max_time_step_reached}, observing concept drift.")
    else:
        st.info(f"Currently displaying monitoring data up to time step {st.session_state.max_time_step_reached}, including concept drift.")

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
        <i>The performance metrics, particularly Accuracy, show a clear decline as concept drift takes effect. This indicates that the model's underlying assumptions are no longer valid.</i>
        </p>
    """, unsafe_allow_html=True)


    # Plotting Drift Metrics (these should ideally remain stable or less affected by concept drift)
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
        <i>While performance degrades significantly, the K-S and JSD statistics for 'Feature 0' remain relatively stable. This distinction is crucial for a Risk Manager, as it helps differentiate concept drift (change in relationship) from data drift (change in input distribution).</i>
        </p>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        Having observed various forms of model degradation and drift, it's clear that continuous monitoring requires pre-defined alert thresholds. Your ability to configure these thresholds appropriately is key to proactive risk management.
        """
    )

    if st.button("Configure Alert Thresholds"):
        st.session_state.current_page = "Configuring Alert Thresholds"
        st.rerun()
