
import streamlit as st
from utils import predict_with_model, calculate_classification_metrics

def main():
    st.markdown(
        """
        ## Reviewing Baseline Performance

        As a Risk Manager, a thorough understanding of the Champion model's initial performance is non-negotiable. These baseline metrics serve as the crucial benchmark against which all future model behavior in production will be evaluated. Any significant deviation from these established levels will trigger alerts and require investigation.

        ### Persona's Action: Establishing Performance Benchmarks
        Your task here is to meticulously review the calculated performance metrics, internalizing what constitutes "normal" and "acceptable" model behavior. This forms the foundation for setting robust monitoring thresholds and risk tolerance levels.

        --- 
        ### Defining and Calculating Classification Performance Metrics
        We will focus on four key metrics essential for a classification model:

        *   **Accuracy:** The proportion of correctly classified instances. It provides a general sense of how well the model is performing across all classes.
            $$A = \frac{TP + TN}{TP + TN + FP + FN}$$

        *   **Precision:** The proportion of positive identifications that were actually correct. This metric is crucial when the cost of false positives is high (e.g., falsely flagging a healthy customer as high risk).
            $$P = \frac{TP}{TP + FP}$$

        *   **Recall (Sensitivity):** The proportion of actual positives that were identified correctly. This is important when the cost of false negatives is high (e.g., failing to identify a fraudulent transaction).
            $$R = \frac{TP}{TP + FN}$$

        *   **F1-Score:** The harmonic mean of Precision and Recall, offering a balanced view of performance, especially useful when there's an uneven class distribution.
            $$F1 = 2 \times \frac{P \times R}{P + R}$$
        Where:
        *   TP: True Positives (correctly predicted positive)
        *   TN: True Negatives (correctly predicted negative)
        *   FP: False Positives (incorrectly predicted positive)
        *   FN: False Negatives (incorrectly predicted negative)
        """
    )

    if 'baseline_metrics' not in st.session_state:
        y_pred_baseline = predict_with_model(st.session_state.champion_model, st.session_state.X_test_baseline)
        st.session_state.baseline_metrics = calculate_classification_metrics(st.session_state.y_test_baseline, y_pred_baseline)

    st.subheader("Champion Model Baseline Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Accuracy",
            value=f"{st.session_state.baseline_metrics['accuracy']:.4f}",
            help="The proportion of correctly classified instances. A drop below the set threshold indicates a performance risk."
        )
    with col2:
        st.metric(
            label="Precision",
            value=f"{st.session_state.baseline_metrics['precision']:.4f}",
            help="The proportion of positive identifications that were actually correct. Crucial when false positives are costly."
        )
    with col3:
        st.metric(
            label="Recall",
            value=f"{st.session_state.baseline_metrics['recall']:.4f}",
            help="The proportion of actual positives that were identified correctly. Important when false negatives are costly."
        )
    with col4:
        st.metric(
            label="F1-Score",
            value=f"{st.session_state.baseline_metrics['f1_score']:.4f}",
            help="The harmonic mean of Precision and Recall, offering a balanced view of performance."
        )

    st.markdown(
        """
        These baseline metrics represent the Champion model's expected performance under ideal, untainted conditions. They provide a critical reference point for continuous monitoring. Any significant drift from these values during production could signal model degradation or data integrity issues, necessitating your immediate attention.

        Now that we have established the performance baseline, it's time to simulate the model's deployment and initiate continuous monitoring to ensure its ongoing health and reliability.
        """
    )

    if st.button("Start Initial Stable Monitoring"):
        st.session_state.current_page = "Initial Stable Monitoring"
        st.rerun()
