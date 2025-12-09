
import streamlit as st
import pandas as pd
import numpy as np
from utils import predict_with_model, calculate_classification_metrics, calculate_data_drift_ks, calculate_data_drift_jsd, compare_champion_challenger

def on_compare_clicked():
    if st.session_state.challenger_model is None:
        st.warning("Please train the Challenger Model first on the previous page.")
        return

    st.info("Comparing Champion and Challenger models...")

    # Use the entire accumulated monitoring data for a comprehensive comparison
    current_monitoring_X = st.session_state.monitoring_data_X.to_numpy()
    current_monitoring_y = st.session_state.monitoring_data_y

    if len(current_monitoring_X) == 0 or len(current_monitoring_y) == 0:
        st.warning("No monitoring data available for comparison.")
        return

    # Evaluate Champion on current monitoring data
    y_pred_champion_recent = predict_with_model(st.session_state.champion_model, current_monitoring_X)
    champion_metrics_recent = calculate_classification_metrics(current_monitoring_y, y_pred_champion_recent)

    # Evaluate Challenger on current monitoring data
    y_pred_challenger_recent = predict_with_model(st.session_state.challenger_model, current_monitoring_X)
    challenger_metrics_recent = calculate_classification_metrics(current_monitoring_y, y_pred_challenger_recent)

    # Calculate drift metrics for the current monitoring data against the original baseline
    # These metrics describe the data itself, so they are the same for Champion and Challenger evaluation
    baseline_feature_0 = pd.DataFrame(st.session_state.baseline_X, columns=[f'feature_{i}' for i in range(st.session_state.baseline_X.shape[1])])['feature_0']
    current_feature_0 = st.session_state.monitoring_data_X['feature_0']

    ks_f0, _ = calculate_data_drift_ks(baseline_feature_0, current_feature_0)
    jsd_f0 = calculate_data_drift_jsd(baseline_feature_0, current_feature_0)

    # Prepare dictionaries for comparison plots (including drift metrics for context if needed by plotting function)
    champion_all_metrics = {**champion_metrics_recent, 'ks_statistic_feature_0': ks_f0, 'jsd_statistic_feature_0': jsd_f0}
    challenger_all_metrics = {**challenger_metrics_recent, 'ks_statistic_feature_0': ks_f0, 'jsd_statistic_feature_0': jsd_f0}

    fig_perf_comp, fig_drift_comp = compare_champion_challenger(champion_metrics_recent, challenger_metrics_recent, champion_all_metrics, challenger_all_metrics)

    st.session_state.fig_perf_comp = fig_perf_comp
    st.session_state.fig_drift_comp = fig_drift_comp
    st.session_state.compare_results_displayed = True
    st.session_state.promote_button_enabled = True # Enable the promote button after comparison
    st.success("Comparison complete. Review the results below.")
    st.rerun()

def on_promote_clicked():
    if st.session_state.challenger_model is not None:
        st.session_state.champion_model = st.session_state.challenger_model
        st.session_state.champion_model_name = "Challenger Model (Promoted)" # Update sidebar status
        st.success("Challenger Model successfully promoted to become the new Champion!")
        st.info("Monitoring will now continue with the new Champion model. Historical monitoring data has been reset to reflect a fresh start.")
        
        # Reset relevant session state for a new monitoring cycle with the new champion
        st.session_state.challenger_model = None
        st.session_state.historical_metrics = []
        st.session_state.monitoring_data_X = pd.DataFrame(columns=[f'feature_{i}' for i in range(st.session_state.baseline_X.shape[1])])
        st.session_state.monitoring_data_y = np.array([])
        st.session_state.max_time_step_reached = 0
        st.session_state.simulation_logs['alerts'] = []
        st.session_state.compare_button_enabled = False
        st.session_state.promote_button_enabled = False
        st.session_state.compare_results_displayed = False

        # Optionally, rerun to reflect immediate change in sidebar, or navigate to monitoring page
        st.session_state.current_page = "Initial Stable Monitoring" # Go back to stable monitoring with new champion
        st.rerun()
    else:
        st.warning("No Challenger model available to promote.")

def main():
    st.markdown(
        """
        ## Champion-Challenger Comparison and Model Promotion

        As a Risk Manager, after a Challenger model has been retrained, your crucial next step is to rigorously compare its performance against the current Champion model. This comparison must be data-driven, using the most recent and relevant production data to assess which model is truly superior. The goal is to ensure that any model promoted to Champion status genuinely mitigates the previously identified risks.

        ### Persona's Action: Making a Data-Driven Deployment Decision
        You are now in the decision-making phase. Based on the comparative analysis of performance and drift metrics, you must decide whether the Challenger model offers sufficient improvement to warrant replacing the current Champion. This is a high-stakes decision that directly impacts the ongoing reliability and trustworthiness of our AI system.
        """
    )

    if st.session_state.get('challenger_model') is None:
        st.warning("Please train the Challenger Model on the 'Retraining a Challenger' page first to proceed with the comparison.")
        return

    st.button("Compare Champion vs. Challenger", on_click=on_compare_clicked, disabled=not st.session_state.get('compare_button_enabled', False))

    if st.session_state.get('compare_results_displayed', False):
        st.markdown("\nAnalysis: Observe how the Challenger model performs against the Champion model on recent, potentially drifted data. A superior Challenger is a candidate for promotion.")
        st.plotly_chart(st.session_state.fig_perf_comp, use_container_width=True)
        st.plotly_chart(st.session_state.fig_drift_comp, use_container_width=True)

        st.markdown(
            """
            The comparison plots provide a clear overview. If the Challenger demonstrates superior performance, especially on the data where the Champion struggled, it indicates that retraining has been effective. Your final decision as a Risk Manager is to authorize its promotion to production.
            """
        )

        st.button("Promote Challenger to Champion", on_click=on_promote_clicked, disabled=not st.session_state.get('promote_button_enabled', False))
    else:
        st.info("Click the 'Compare Champion vs. Challenger' button to evaluate the models.")

    if st.button("Proceed to Human-in-the-Loop & Governance"):
        st.session_state.current_page = "Human-in-the-Loop & Governance"
        st.rerun()
