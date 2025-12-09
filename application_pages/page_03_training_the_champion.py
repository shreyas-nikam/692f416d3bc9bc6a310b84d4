
import streamlit as st
from sklearn.model_selection import train_test_split
from utils import train_logistic_regression_model

def main():
    st.markdown(
        """
        ## Training the Champion Model

        With the baseline data established and reviewed, the next logical step for a Risk Manager is to oversee the training of the initial model, which we will designate as our "Champion" model. This model represents the version currently deemed fit for production, and its performance will be continuously monitored.

        ### Persona's Action: Initiating Model Deployment Preparation
        Your role here is to ensure that the model is trained on the approved baseline data, setting the stage for its deployment. This is a critical step in the AI model lifecycle, as the Champion model will serve as the benchmark for all subsequent evaluations and challenger comparisons.
        """
    )

    if 'champion_model' not in st.session_state:
        X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline = train_test_split(
            st.session_state.baseline_X, st.session_state.baseline_y, test_size=0.3, random_state=42
        )
        st.session_state.champion_model = train_logistic_regression_model(X_train_baseline, y_train_baseline, random_state=42)
        st.session_state.X_test_baseline = X_test_baseline # Store for baseline metrics
        st.session_state.y_test_baseline = y_test_baseline
        st.success("Champion Model trained successfully.")
    else:
        st.info("Champion Model is already trained and ready.")

    st.markdown(
        """
        The Champion model has been successfully trained on our baseline dataset. Now, it's imperative to review its initial performance to set realistic expectations for its behavior in production. This involves calculating key classification metrics that will guide our monitoring efforts.
        """
    )

    if st.button("Review Champion Model's Baseline Performance"):
        st.session_state.current_page = "Reviewing Baseline Performance"
        st.rerun()
