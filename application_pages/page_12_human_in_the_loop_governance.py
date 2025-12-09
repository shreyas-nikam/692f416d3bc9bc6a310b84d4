
import streamlit as st
from utils import log_manual_review

def on_manual_review_clicked():
    current_alerts_state = st.session_state.historical_metrics[-1]['alerts'] if st.session_state.historical_metrics else "No recent alerts"
    log_message = log_manual_review(st.session_state.max_time_step_reached, current_alerts_state)
    st.info(f"Manual review triggered. A risk manager would now investigate and decide on appropriate actions.")
    st.rerun() # Rerun to update the text area immediately

def main():
    st.markdown(
        """
        ## Human-in-the-Loop and Governance in AI Monitoring (SR 11-7)

        As a Risk Manager, while automated monitoring and model retraining are powerful tools, they are not a substitute for human oversight. Especially for critical AI models in regulated environments, a "human-in-the-loop" (HITL) approach is essential. This ensures that complex alerts, ambiguous situations, or high-impact decisions are subjected to qualitative assessment, expert judgment, and adherence to governance protocols, as emphasized by regulatory guidance like SR 11-7.

        ### Persona's Action: Exercising Prudent Oversight
        Your role is to activate manual review processes when automated systems signal issues that require deeper investigation or strategic decisions. This demonstrates the critical balance between automation and human intelligence in maintaining trustworthy AI. It's about ensuring that accountability remains firmly with human experts, especially when model decisions have significant consequences.
        """
    )

    st.subheader("Trigger a Manual Review Event")
    st.markdown(
        """
        In a real-world scenario, you might trigger a manual review when an alert is ambiguous, when a new type of drift is observed, or when regulatory scrutiny requires explicit human validation of a model's state or a promotion decision.
        """
    )

    if st.button("Trigger Manual Review", on_click=on_manual_review_clicked):
        pass # Logic handled by the on_manual_review_clicked function

    st.subheader("Manual Review Log")
    if 'manual_review_log' in st.session_state and st.session_state.manual_review_log:
        st.text_area("Recorded Manual Review Events", value="\n".join(st.session_state.manual_review_log), height=150, disabled=True)
    else:
        st.info("No manual review events recorded yet.")

    st.markdown(
        """
        This page underscores that even with the most sophisticated monitoring and automated interventions, human judgment and governance remain indispensable in the AI model lifecycle. Your final responsibility is to ensure that these processes are robust and auditable.
        """
    )

    if st.button("Proceed to Conclusion"):
        st.session_state.current_page = "Conclusion"
        st.rerun()
