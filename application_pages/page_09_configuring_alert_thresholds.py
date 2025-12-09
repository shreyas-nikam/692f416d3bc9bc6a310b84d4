
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import run_monitoring_step # Needed if we want to run a new step, but here we re-evaluate historical

def update_alert_thresholds():
    # This function is called when slider values change
    st.session_state.current_alert_thresholds['accuracy_min'] = st.session_state.accuracy_threshold_slider
    st.session_state.current_alert_thresholds['ks_max'] = st.session_state.ks_threshold_slider
    st.session_state.current_alert_thresholds['jsd_max'] = st.session_state.jsd_threshold_slider

    # Re-evaluate all historical_metrics with new thresholds
    updated_alerts_log = []
    for i, step_data in enumerate(st.session_state.historical_metrics):
        re_evaluated_alerts = {}
        current_metrics = step_data['metrics']
        time_step = step_data['time_step']

        if current_metrics['accuracy'] < st.session_state.current_alert_thresholds['accuracy_min']:
            re_evaluated_alerts['accuracy'] = f"Accuracy ({current_metrics['accuracy']:.4f}) below threshold ({st.session_state.current_alert_thresholds['accuracy_min']:.4f})."
        if current_metrics['ks_statistic_feature_0'] > st.session_state.current_alert_thresholds['ks_max']:
            re_evaluated_alerts['ks_feature_0'] = f"K-S Stat (Feature 0) ({current_metrics['ks_statistic_feature_0']:.4f}) above threshold ({st.session_state.current_alert_thresholds['ks_max']:.4f})."
        if current_metrics['jsd_statistic_feature_0'] > st.session_state.current_alert_thresholds['jsd_max']:
            re_evaluated_alerts['jsd_feature_0'] = f"JSD (Feature 0) ({current_metrics['jsd_statistic_feature_0']:.4f}) above threshold ({st.session_state.current_alert_thresholds['jsd_max']:.4f})."
        
        st.session_state.historical_metrics[i]['alerts'] = re_evaluated_alerts
        if re_evaluated_alerts:
            alert_details_str = ", ".join(re_evaluated_alerts.values())
            updated_alerts_log.append(f"ALERT at Time Step {time_step}: {alert_details_str}")
    
    st.session_state.simulation_logs['alerts'] = updated_alerts_log # Update log for display
    st.rerun() # Rerun to re-render plots with new thresholds and highlighted alerts

def main():
    st.markdown(
        """
        ## Configuring Alert Thresholds

        As a Risk Manager, your ability to define and adjust alert thresholds is central to proactive AI model risk management. These thresholds are not arbitrary; they reflect the organization's risk appetite and the criticality of the model's function. Setting them appropriately ensures that genuine issues are flagged without creating excessive noise from false positives.

        ### Persona's Action: Defining Risk Boundaries
        In this step, you will actively configure the boundaries for acceptable model performance and data stability. Adjusting these sliders allows you to immediately see how different risk tolerances would have impacted the detection of issues during the simulated monitoring period. This hands-on experience reinforces the practical implications of your risk management decisions.
        """
    )

    st.subheader("Adjust Alert Thresholds")

    # Sliders for threshold adjustment
    st.slider(
        label=r"Min. Accuracy Threshold $\Delta A_{min}$",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.current_alert_thresholds['accuracy_min'],
        step=0.01,
        key="accuracy_threshold_slider",
        on_change=update_alert_thresholds,
        help="Adjust this slider to set the minimum acceptable accuracy score for the model. A lower value indicates higher risk tolerance."
    )

    st.slider(
        label=r"Max. K-S Statistic Threshold (Feature 0) $\Delta KS_{max}$",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.current_alert_thresholds['ks_max'],
        step=0.01,
        key="ks_threshold_slider",
        on_change=update_alert_thresholds,
        help="Set the maximum tolerable difference in feature distributions as measured by the K-S test. Higher values mean more drift."
    )

    st.slider(
        label=r"Max. JSD Threshold (Feature 0) $\Delta JSD_{max}$",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.current_alert_thresholds['jsd_max'],
        step=0.01,
        key="jsd_threshold_slider",
        on_change=update_alert_thresholds,
        help="Set the maximum tolerable dissimilarity between feature distributions using Jensen-Shannon Divergence. Higher values mean greater divergence."
    )

    st.markdown(
        """
        --- 
        ### Monitoring Dashboard with Dynamic Thresholds
        Below are the updated performance and drift plots. As you adjust the thresholds above, observe how the horizontal alert lines shift, and how previously missed or newly triggered alerts appear. This interactive feedback helps you fine-tune your monitoring strategy.
        """
    )

    # Extract data for plotting
    time_steps = [d['time_step'] for d in st.session_state.historical_metrics]
    accuracies = [d['metrics']['accuracy'] for d in st.session_state.historical_metrics]
    precisions = [d['metrics']['precision'] for d in st.session_state.historical_metrics]
    recal_scores = [d['metrics']['recall'] for d in st.session_state.historical_metrics]
    f1_scores = [d['metrics']['f1_score'] for d in st.session_state.historical_metrics]
    ks_stats = [d['metrics']['ks_statistic_feature_0'] for d in st.session_state.historical_metrics]
    jsd_stats = [d['metrics']['jsd_statistic_feature_0'] for d in st.session_state.historical_metrics]

    # Identify points where alerts were triggered based on current thresholds
    alert_accuracy_x, alert_accuracy_y = [], []
    alert_ks_x, alert_ks_y = [], []
    alert_jsd_x, alert_jsd_y = [], []

    for d in st.session_state.historical_metrics:
        if 'accuracy' in d['alerts']:
            alert_accuracy_x.append(d['time_step'])
            alert_accuracy_y.append(d['metrics']['accuracy'])
        if 'ks_feature_0' in d['alerts']:
            alert_ks_x.append(d['time_step'])
            alert_ks_y.append(d['metrics']['ks_statistic_feature_0'])
        if 'jsd_feature_0' in d['alerts']:
            alert_jsd_x.append(d['time_step'])
            alert_jsd_y.append(d['metrics']['jsd_statistic_feature_0'])

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

    # Add triggered accuracy alerts as distinct markers
    if alert_accuracy_x:
        fig_perf.add_trace(go.Scatter(x=alert_accuracy_x, y=alert_accuracy_y, mode='markers', name='Accuracy Alert',
                                      marker=dict(color='red', size=10, symbol='x'),
                                      hovertemplate='Time Step: %{x}<br>Accuracy Alert: %{y:.4f}<extra></extra>'))

    # Add accuracy threshold line
    fig_perf.add_hline(y=st.session_state.current_alert_thresholds['accuracy_min'], line_dash="dash", line_color="red",
                       annotation_text=f"Min. Accuracy Threshold ({st.session_state.current_alert_thresholds['accuracy_min']:.2f})",
                       annotation_position="bottom right")

    fig_perf.update_layout(title='Model Performance Metrics Over Time (with Alerts)',
                           xaxis_title='Time Step',
                           yaxis_title='Score',
                           hovermode="x unified")
    st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown(r"""
        <p style="font-size: smaller; text-align: center; color: grey;">
        <i>The performance metrics plot now displays the dynamically adjustable minimum accuracy threshold (red dashed line). Any data points (X markers) falling below this line indicate a triggered alert, reflecting your configured risk tolerance.</i>
        </p>
    """, unsafe_allow_html=True)

    # Plotting Drift Metrics
    fig_drift = go.Figure()
    fig_drift.add_trace(go.Scatter(x=time_steps, y=ks_stats, mode='lines+markers', name='K-S Statistic (Feature 0)',
                                   hovertemplate='Time Step: %{x}<br>K-S Stat: %{y:.4f}<extra></extra>'))
    fig_drift.add_trace(go.Scatter(x=time_steps, y=jsd_stats, mode='lines+markers', name='JSD (Feature 0)',
                                   hovertemplate='Time Step: %{x}<br>JSD: %{y:.4f}<extra></extra>'))

    # Add triggered drift alerts as distinct markers
    if alert_ks_x:
        fig_drift.add_trace(go.Scatter(x=alert_ks_x, y=alert_ks_y, mode='markers', name='K-S Alert',
                                      marker=dict(color='orange', size=10, symbol='circle-x'),
                                      hovertemplate='Time Step: %{x}<br>K-S Alert: %{y:.4f}<extra></extra>'))
    if alert_jsd_x:
        fig_drift.add_trace(go.Scatter(x=alert_jsd_x, y=alert_jsd_y, mode='markers', name='JSD Alert',
                                      marker=dict(color='purple', size=10, symbol='diamond-x'),
                                      hovertemplate='Time Step: %{x}<br>JSD Alert: %{y:.4f}<extra></extra>'))

    # Add K-S and JSD thresholds
    fig_drift.add_hline(y=st.session_state.current_alert_thresholds['ks_max'], line_dash="dash", line_color="orange",
                        annotation_text=f"Max. K-S Threshold ({st.session_state.current_alert_thresholds['ks_max']:.2f})",
                        annotation_position="bottom right")
    fig_drift.add_hline(y=st.session_state.current_alert_thresholds['jsd_max'], line_dash="dash", line_color="purple",
                        annotation_text=f"Max. JSD Threshold ({st.session_state.current_alert_thresholds['jsd_max']:.2f})",
                        annotation_position="top right")

    fig_drift.update_layout(title='Data Drift Metrics Over Time (Feature 0) (with Alerts)',
                            xaxis_title='Time Step',
                            yaxis_title='Drift Statistic',
                            hovermode="x unified")
    st.plotly_chart(fig_drift, use_container_width=True)

    st.markdown(r"""
        <p style="font-size: smaller; text-align: center; color: grey;">
        <i>The data drift metrics plot dynamically displays the maximum K-S (orange dashed line) and JSD (purple dashed line) thresholds. Data points (X markers for K-S, diamond-X for JSD) exceeding these lines indicate triggered data drift alerts.</i>
        </p>
    """, unsafe_allow_html=True)

    st.subheader("Alert Log")
    st.text_area("Alert Events Log", value="\n".join(st.session_state.simulation_logs['alerts']), height=200, disabled=True)

    st.markdown(
        """
        By interactively setting these thresholds, you gain a deeper appreciation for the balance between sensitivity to change and avoiding alert fatigue. Now that you have configured your monitoring, it's time to consider intervention strategies when alerts are triggered.
        """
    )

    if st.button("Proceed to Retrain a Challenger Model"):
        st.session_state.current_page = "Retraining a Challenger"
        st.rerun()
