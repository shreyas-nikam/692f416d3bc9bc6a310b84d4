
import streamlit as st
import pandas as pd
import numpy as np
from utils import retrain_challenger_model, predict_with_model, calculate_classification_metrics

def on_retrain_clicked():
    st.info("Simulating model retraining...")
    
    # Combine baseline data with all accumulated monitoring data for challenger training
    challenger_X_train = pd.concat([
        pd.DataFrame(st.session_state.baseline_X, columns=[f'feature_{i}' for i in range(st.session_state.baseline_X.shape[1])]),
        st.session_state.monitoring_data_X
    ], ignore_index=True)
    challenger_y_train = np.concatenate([st.session_state.baseline_y, st.session_state.monitoring_data_y])
    
    st.session_state.challenger_model = retrain_challenger_model(challenger_X_train, challenger_y_train, random_state=42)
    st.success("Challenger Model trained successfully.")
    
    # Evaluate Challenger's performance on the *most recent* monitoring data
    # We need to ensure monitoring_data_X has enough samples for evaluation, so use the last batch
    # Assuming num_samples_per_step = 100 as in run_monitoring_step
    num_samples_per_step = 100 # Adjust if run_monitoring_step changes
    recent_X = st.session_state.monitoring_data_X.tail(num_samples_per_step).to_numpy()
    recent_y = st.session_state.monitoring_data_y[-num_samples_per_step:]

    if len(recent_X) > 0 and len(recent_y) > 0:
        y_pred_challenger_recent = predict_with_model(st.session_state.challenger_model, recent_X)
        challenger_metrics_recent = calculate_classification_metrics(recent_y, y_pred_challenger_recent)
        st.subheader("Challenger Model Performance on Recent Monitoring Data:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{challenger_metrics_recent['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{challenger_metrics_recent['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{challenger_metrics_recent['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{challenger_metrics_recent['f1_score']:.4f}")
    else:
        st.warning("Not enough recent monitoring data to evaluate Challenger's performance.")

    st.session_state.compare_button_enabled = True # Enable next button
    st.rerun()

def main():
    st.markdown(
        """
        ## Simulating Interventions: Retraining a "Challenger" Model

        As a Risk Manager, detecting performance degradation or data drift is only the first step. The next critical action is intervention. When monitoring alerts signal that a Champion model is no longer fit for purpose, the standard protocol is to retrain a new model, often referred to as a "Challenger." This new model is trained on up-to-date data, including the recent production data that caused the Champion's issues.

        ### Persona's Action: Initiating a Corrective Action
        Your role here is to trigger the retraining process. This decision is informed by the insights gathered from continuous monitoring and alert configurations. The Challenger model represents a potential solution to mitigate the risks associated with the Champion's declining health. Once trained, it will be rigorously compared against the current Champion to determine its suitability for deployment.
        """
    )

    if st.button("Simulate Retraining Challenger Model", on_click=on_retrain_clicked):
        pass # The action is handled by on_retrain_clicked

    if 'challenger_model' in st.session_state and st.session_state.challenger_model is not None:
        st.success("Challenger Model has been trained. You can now compare it with the Champion.")
        if st.session_state.compare_button_enabled:
            if st.button("Proceed to Champion-Challenger Comparison"):
                st.session_state.current_page = "Champion-Challenger Comparison"
                st.rerun()
    else:
        st.info("Click the button above to train a new Challenger model.")
