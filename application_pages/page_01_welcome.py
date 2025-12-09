
import streamlit as st

def main():
    st.markdown(
        """
        As a **Risk Manager**, your primary responsibility is to ensure the reliability, fairness, and compliance of AI models deployed within our financial institution. The dynamic nature of AI in production, coupled with evolving data landscapes, presents continuous challenges that necessitate proactive model risk management.

        This simulator will guide you through a realistic scenario where you oversee a critical AI model, confronting issues such as performance degradation and data drift. Our goal is to equip you with the practical skills to apply Model Risk Management (MRM) principles, particularly those adapted from regulatory frameworks like SR 11-7, to real-world challenges.

        ### Your Mission:
        Throughout this journey, you will:
        -   **Establish a Baseline:** Understand the model's initial training data and expected performance.
        -   **Deploy a Champion Model:** Simulate putting a model into production.
        -   **Initiate Monitoring:** Observe the model's health metrics over simulated time steps.
        -   **Encounter Degradation & Drift:** Witness the model's performance decline due to various factors (performance degradation, data drift, concept drift).
        -   **Proactively Manage Risk:** Configure alert thresholds to flag issues promptly.
        -   **Intervene:** Trigger the retraining of a new "Challenger" model.
        -   **Evaluate & Promote:** Compare the Challenger against the Champion and decide on deployment.
        -   **Ensure Governance:** Simulate human oversight for critical decisions.

        This application provides a hands-on, interactive experience, allowing you to visualize model behavior, experiment with controls, gain insights into different types of model degradation, and reinforce essential MRM principles.

        Ready to ensure the trustworthiness of our AI models?
        """
    )

    if st.button("Start My Monitoring Journey"):
        st.session_state.current_page = "Establishing the Baseline"
        st.rerun()
