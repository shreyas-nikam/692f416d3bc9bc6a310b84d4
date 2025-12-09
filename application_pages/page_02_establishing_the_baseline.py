
import streamlit as st
import pandas as pd
import numpy as np
from utils import generate_synthetic_classification_data

def main():
    st.markdown(
        """
        ## Establishing the Baseline

        Before deploying any AI model into production, it is crucial for a Risk Manager to thoroughly understand its foundational characteristics. This includes defining the initial training data, its distribution, and the expected performance benchmarks. This "baseline" serves as the standard against which all future model performance and data shifts will be measured.

        In this step, we will simulate the generation of a synthetic dataset that represents the kind of data our AI model (a binary classifier in this case) would be trained on. This dataset will inform our understanding of the model's initial learning environment.

        ### Persona's Action: Reviewing Initial Data Integrity
        As a Risk Manager, your first task is to ensure the integrity and representativeness of the data used for model training. Understanding the features and target distribution helps you anticipate potential biases or imbalances, which are critical inputs for your initial risk assessment.
        """
    )

    # Initial generation, run once if not in session state
    if 'baseline_X' not in st.session_state:
        st.session_state.baseline_X, st.session_state.baseline_y = generate_synthetic_classification_data(
            num_samples=1000, num_features=5, n_informative=3, n_redundant=0,
            n_clusters_per_class=1, random_state=42
        )
        st.session_state.baseline_df = pd.DataFrame(st.session_state.baseline_X, columns=[f'feature_{i}' for i in range(st.session_state.baseline_X.shape[1])])
        st.session_state.baseline_df['target'] = st.session_state.baseline_y

    st.subheader("Glimpse of Baseline Data")
    st.dataframe(st.session_state.baseline_df.head())
    st.write(f"Baseline Data Shape: {st.session_state.baseline_df.shape}")
    st.write(f"Baseline Target Distribution:\n{st.session_state.baseline_df['target'].value_counts(normalize=True)}")

    st.markdown(
        """
        The output above shows a glimpse of the synthetic baseline data, including its shape and the distribution of the target variable. This initial overview confirms that the data is structured as expected and provides the necessary foundation for training our Champion model.

        Having verified the baseline data, the next step in your workflow is to oversee the training of the initial model that will be deployed.
        """
    )

    if st.button("Proceed to Train the Champion Model"):
        st.session_state.current_page = "Training the Champion"
        st.rerun()
