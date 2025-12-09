
import streamlit as st

def on_restart_simulation_clicked():
    # Clear all session state variables to restart the application
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state.current_page = "Welcome & Your Role" # Redirect to the first page
    st.rerun()

def main():
    st.markdown(
        """
        ## Conclusion: Continuous Monitoring for Trustworthy AI

        Congratulations, Risk Manager! You have successfully navigated the dynamic landscape of AI model health, from establishing baselines and detecting various forms of drift and degradation to configuring alerts and orchestrating interventions. This simulated journey has reinforced the critical importance of a robust AI Model Risk Management (MRM) framework, particularly one adapted from principles like SR 11-7.

        ### Key Takeaways for Your Role:
        -   **Proactive Detection:** Continuous monitoring is not merely reactive; it's about proactively identifying subtle shifts before they escalate into significant risks.
        -   **Adaptive Controls:** Alert thresholds are dynamic and should reflect the evolving risk appetite and model criticality. Your ability to adjust them is a key control mechanism.
        -   **Champion-Challenger Framework:** The iterative process of retraining and comparing models ensures that your organization can adapt to new data environments and maintain optimal performance.
        -   **Human-in-the-Loop:** Even with advanced automation, human expertise and judgment are irreplaceable for critical decisions, policy enforcement, and addressing nuanced risks.
        -   **Governance and Accountability:** Every step, from data validation to model promotion, requires clear governance protocols and auditable records to ensure compliance and build trust.

        By applying these principles, you contribute directly to building and maintaining trustworthy and reliable AI systems within our financial institution. Your vigilance is a cornerstone of responsible AI deployment.

        Thank you for your dedication to sound AI Model Risk Management!
        """
    )

    st.button("Restart Simulation", on_click=on_restart_simulation_clicked)
