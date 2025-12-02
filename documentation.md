id: 692f416d3bc9bc6a310b84d4_documentation
summary: AI Design and Deployment Lab 4 Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# AI Model Health Dashboard with Streamlit for Model Risk Management (MRM)

## 1. Introduction to AI Model Risk Management (MRM) and the Dashboard
Duration: 0:05

Welcome to this codelab on building and understanding an AI Model Health Dashboard using Streamlit. In today's data-driven world, AI models are increasingly deployed in critical business operations, especially in regulated industries like finance. This makes **Model Risk Management (MRM)** an indispensable practice. MRM, as highlighted by regulatory guidelines such as **SR 11-7** from the Federal Reserve, ensures that models are robust, reliable, and perform as expected throughout their lifecycle, mitigating potential financial, reputational, and operational risks.

This interactive Streamlit application serves as a powerful simulator to demonstrate key MRM principles. It provides a comprehensive guide for developers and risk managers to:

*   **Understand Model Health**: Gain insights into crucial metrics that define the operational health of an AI model in a production environment.
*   **Detect Drift**: Learn to identify various forms of model degradation, including data drift ($P(X)$ changes) and concept drift ($P(Y|X)$ changes), which are common challenges for adaptive AI systems.
*   **Implement Monitoring**: Set up continuous monitoring pipelines to track model performance and data characteristics over time.
*   **Strategize Interventions**: Explore effective strategies like model retraining, Champion-Challenger comparisons, and human-in-the-loop processes to address detected issues.
*   **Align with Regulations**: Reinforce the practical application of SR 11-7 principles for ongoing validation and risk control.

### Application Architecture Overview

The Streamlit application is structured for clarity and modularity. Here's a high-level overview of its components and how they interact:

1.  **`app.py` (Main Entry Point)**:
    *   Initializes Streamlit page configuration.
    *   Includes all utility functions (data generation, model training, metric calculation, drift detection) for global availability.
    *   Sets up the navigation for different pages (currently only "Dashboard").
    *   Calls the `main` function from `application_pages/dashboard.py` to render the primary dashboard interface.
2.  **`application_pages/dashboard.py` (Dashboard Logic)**:
    *   Contains the `main` function responsible for rendering the Streamlit UI elements for the dashboard.
    *   Manages the application's state using `st.session_state`.
    *   Orchestrates the simulation steps: baseline setup, continuous monitoring, and intervention strategies.
    *   Utilizes the utility functions (defined at the top of `app.py` for global access) to perform operations like data generation, model training, metric calculation, and plotting.
3.  **Utility Functions**: These functions, present at the top of `app.py`, encapsulate core logic:
    *   `generate_synthetic_classification_data`: Creates simulated datasets with controllable drift.
    *   `train_logistic_regression_model`: Trains a simple Logistic Regression model.
    *   `calculate_classification_metrics`: Computes standard performance metrics.
    *   `calculate_data_drift_ks`, `calculate_data_drift_jsd`: Measures data drift using statistical methods.
    *   `run_monitoring_step`: Executes one step of the monitoring loop.
    *   `compare_champion_challenger`, `log_manual_review`, `init_session_state`: Support intervention and state management.

This architecture ensures that the core logic is reusable and the UI remains responsive and organized.

## 2. Exploring the Application's Core Code Structure
Duration: 0:10

Before diving into the interactive dashboard, let's explore the Python code that powers it. Understanding the underlying functions and how Streamlit manages state and caching is crucial for developers.

### `app.py` - The Streamlit Entry Point

The `app.py` file is where our Streamlit application begins. It handles the initial setup and delegates to the dashboard logic.

```python
# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
import plotly.graph_objects as go

# All functions from dashboard.py are copied here for global availability
# ... (function definitions as provided in the snippet) ...

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

st.markdown("""
# ... introductory text ...
""")

page = st.sidebar.selectbox(label="Navigation", options=["Dashboard"])

if page == "Dashboard":
    from application_pages.dashboard import main
    main()
```

<aside class="positive">
<b>Key Takeaway:</b> Notice how `app.py` imports `main` from `application_pages.dashboard` and then calls `main()`. This modular approach keeps the main application logic separate from the Streamlit entry point, making the code more organized and easier to maintain. All helper functions are also placed at the top of `app.py` for direct access by the `dashboard.main` function.
</aside>

### Streamlit Caching (`@st.cache_data`, `@st.cache_resource`)

Several functions are decorated with `@st.cache_data` or `@st.cache_resource`. These decorators are powerful Streamlit features that optimize performance by caching function outputs.

*   `@st.cache_data(ttl="2h")`: Caches dataframes, lists, dicts, or any data. The `ttl` (time-to-live) parameter specifies how long the cached data remains valid. If a function with this decorator is called again with the same inputs, Streamlit returns the cached result instead of re-executing the function. This is perfect for data generation and metric calculations that are computationally intensive but produce static results for given inputs.
*   `@st.cache_resource`: Caches global resources like machine learning models, database connections, or other objects that shouldn't be reloaded or re-initialized on every rerun. This is used for our `train_logistic_regression_model` function to ensure the model isn't retrained unless its inputs change.

```python
@st.cache_data(ttl="2h")
def generate_synthetic_classification_data(...):
    # ... data generation logic ...

@st.cache_resource
def train_logistic_regression_model(X_train, y_train, random_state=42):
    # ... model training logic ...
```

<aside class="positive">
<b>Best Practice:</b> Using caching decorators significantly improves the user experience in Streamlit apps by preventing unnecessary re-computation, especially in interactive dashboards where user inputs can trigger frequent reruns.
</aside>

### Session State (`st.session_state`)

The application extensively uses `st.session_state` to store and persist variables across Streamlit reruns. This is fundamental for maintaining the state of our simulation, such as the trained models, historical metrics, and drift parameters.

```python
def init_session_state():
    if 'baseline_X' not in st.session_state:
        st.session_state.baseline_X = None
        st.session_state.baseline_y = None
        # ... initialize other session state variables ...
```

The `init_session_state()` function ensures that all necessary variables are initialized the first time the application runs or when the user clears the state. This includes:
*   `baseline_X`, `baseline_y`: The initial dataset.
*   `champion_model`, `challenger_model`: The trained models.
*   `historical_accuracy`, `historical_ks_stats_f0`, etc.: Lists to store monitoring results over time.
*   `drift_parameters`, `alert_thresholds`: User-configurable simulation settings.

<aside class="positive">
<b>Developer Tip:</b> `st.session_state` is crucial for building interactive Streamlit applications. Without it, variables would reset every time a widget is interacted with, making complex simulations impossible.
</aside>

### Core Utility Functions

Review the following key utility functions in `app.py` (which are also replicated in `application_pages/dashboard.py` in the provided setup, but for development, consider them as globally available functions):

*   **`generate_synthetic_classification_data(...)`**: Creates synthetic data for binary classification. It can also inject different types of "drift":
    *   `mean_shift`, `std_factor`: For data drift (change in $P(X)$).
    *   `concept_drift_factor`: For concept drift (change in $P(Y|X)$).
    *   `performance_degradation_factor`: For general label noise, simulating performance decay.
*   **`train_logistic_regression_model(X_train, y_train, random_state=42)`**: Trains a `LogisticRegression` model from `sklearn`.
*   **`calculate_classification_metrics(y_true, y_pred)`**: Returns a dictionary of Accuracy, Precision, Recall, and F1-Score.
*   **`calculate_data_drift_ks(baseline_feature_data, current_feature_data)`**: Computes the Kolmogorov-Smirnov (K-S) statistic.
*   **`calculate_data_drift_jsd(baseline_feature_data, current_feature_data, num_bins=50)`**: Computes the Jensen-Shannon Divergence (JSD).
*   **`run_monitoring_step(...)`**: This is the heart of the simulation, generating a new batch of data, evaluating the champion model, calculating drift, and checking for alerts.
*   **`compare_champion_challenger(...)`**: Facilitates visual and quantitative comparison between two models.

These functions abstract the complexity of data science tasks, allowing the Streamlit UI in `dashboard.py` to focus on presentation and user interaction.

## 3. Setting Up the Baseline and Initial Champion Model
Duration: 0:15

This step walks you through the initial setup of the Model Health Dashboard. We will generate synthetic baseline data and train our first "Champion" model, which will serve as the reference for all subsequent monitoring.

Navigate to the "Dashboard" page in the Streamlit application.

### Understanding the Baseline Section

The first section of the dashboard is titled "1. Baseline Setup".

```python
# From application_pages/dashboard.py main function
st.subheader("1. Baseline Setup")
st.markdown(r"""
In this section, we set up our initial environment by generating synthetic baseline data and training our first "Champion" model.
This baseline will serve as the reference point for all subsequent monitoring and drift detection.
""")

with st.expander("Generate Baseline Data & Train Champion Model"):
    st.markdown("Configure parameters for synthetic data generation:")
    # ... input widgets for num_samples, num_features, etc. ...

    if st.button("Generate Baseline Data and Train Initial Champion Model"):
        # ... logic to call generate_synthetic_classification_data and train_logistic_regression_model ...
```

This section allows you to configure the parameters for generating the synthetic dataset:
*   **Number of Baseline Samples**: The size of your initial training data.
*   **Number of Features**: How many input features your model will use.
*   **Number of Informative Features**: Features that are actually useful for predicting the target.
*   **Number of Redundant Features**: Features that are linear combinations of informative features.
*   **Clusters per Class**: Controls the complexity of the data distribution.
*   **Random State**: For reproducibility of the synthetic data generation.

### Action: Generate Baseline Data and Train Champion Model

1.  **Open the "Generate Baseline Data & Train Champion Model" expander.**
2.  **Adjust the parameters if you wish.** For this codelab, you can use the default values (e.g., 1000 samples, 10 features, 3 informative features).
3.  **Click the "Generate Baseline Data and Train Initial Champion Model" button.**

<aside class="positive">
You should see success messages indicating that the baseline data has been generated and the initial Champion Model has been trained. The dashboard will then display a snapshot of the baseline data and its performance metrics.
</aside>

Let's look at the core logic behind this action:

```python
# Inside the st.button block in dashboard.py's main function
if st.button("Generate Baseline Data and Train Initial Champion Model"):
    st.session_state.baseline_X, st.session_state.baseline_y = generate_synthetic_classification_data(
        num_samples=num_samples, num_features=num_features, n_informative=n_informative,
        n_redundant=n_redundant, n_clusters_per_class=n_clusters_per_class, random_state=random_state
    )
    st.session_state.baseline_df = pd.DataFrame(st.session_state.baseline_X, columns=[f'feature_{i}' for i in range(num_features)])
    st.session_state.baseline_df['target'] = st.session_state.baseline_y
    st.success("Baseline data generated successfully!")

    # Train initial Champion model
    X_train, X_test, y_train, y_test = train_test_split(st.session_state.baseline_X, st.session_state.baseline_y, test_size=0.0, random_state=random_state)
    # Note: test_size=0.0 means all data is used for training here, for simplicity in a simulator setting.
    # In a real scenario, you'd use a proper test set.
    st.session_state.champion_model = train_logistic_regression_model(X_train, y_train, random_state)
    # For baseline metrics, let's re-split baseline data to get a proper evaluation
    X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(st.session_state.baseline_X, st.session_state.baseline_y, test_size=0.2, random_state=random_state)
    y_pred_baseline = predict_with_model(st.session_state.champion_model, X_test_eval)
    st.session_state.baseline_metrics = calculate_classification_metrics(y_test_eval, y_pred_baseline)
    st.session_state.initial_champion_trained = True
    st.success("Initial Champion Model trained successfully!")

    # Reset monitoring data and history for a fresh start
    st.session_state.historical_accuracy = [st.session_state.baseline_metrics['accuracy']]
    # ... other historical metrics initialization ...
    st.session_state.current_t_step = 0
```

<aside class="negative">
<b>Important Note:</b> In the provided application, the `train_test_split` for initial Champion model training uses `test_size=0.2` for evaluating `baseline_metrics` on a holdout set, which is good. However, when training the model itself, it uses `test_size=0.0` which means the *entire* baseline data is used for training the champion model. For a more robust simulation, a dedicated training split and test split should be used, or a separate validation dataset for champion evaluation. For simplicity in this simulator, it's acceptable.
</aside>

### Reviewing Baseline Information

After the baseline is set up, you will see:

*   **Baseline Data Snapshot**: A preview of the synthetic data used for training.
*   **Baseline Model Performance Metrics**: Key classification metrics (Accuracy, Precision, Recall, F1-Score) for the Champion model on a holdout set of the baseline data. These values represent the expected performance under normal operating conditions.
*   **Baseline Target Distribution**: The proportion of each class in the target variable, indicating if the dataset is balanced.

These initial metrics are crucial for establishing the "normal" behavior of your model. Any significant deviation from these baseline values in later monitoring steps will trigger alerts or require intervention.

## 4. Continuous Monitoring of Model Performance and Drift
Duration: 0:20

This is the core of the dashboard, simulating a production environment where new data arrives, and the model's health is continuously assessed. We will monitor model performance and detect various types of drift.

Make sure you have completed Step 3 and have an initial Champion model trained.

### Introduction to Monitoring

The "2. Continuous Monitoring & Alerts" section provides controls and visualizations for the monitoring process.

```python
# From application_pages/dashboard.py main function
st.subheader("2. Continuous Monitoring & Alerts")
st.markdown(r"""
In this section, we simulate a continuous production environment where new data batches arrive over time.
The dashboard monitors the Champion model's performance and detects data or concept drift.
""")
```

### Configure Simulation Parameters (Introducing Drift)

Within the "Configure Simulation Parameters" expander, you can introduce different types of degradation and drift into the incoming data stream. This is where you can actively experiment with how your model reacts to changes in its environment.

*   **Mean Shift (Feature 0 - Data Drift)**: Shifts the mean of `feature_0`. This simulates a change in the distribution of an input feature ($P(X)$ change).
*   **Std Dev Factor (Feature 0 - Data Drift)**: Multiplies the standard deviation of `feature_0`. This also simulates a change in $P(X)$.
*   **Concept Drift Factor**: Artificially flips a proportion of labels for data points where `feature_0` is above its median. This simulates **concept drift**, where the relationship between input features and the target variable changes ($P(Y|X)$ change).
*   **Performance Degradation Factor**: Randomly flips a proportion of labels across the dataset. This simulates general performance degradation due to unmodeled external factors or increasing noise in labels.

### Action: Introduce Drift and Run Simulation Steps

1.  **Open the "Configure Simulation Parameters" expander.**
2.  **Experiment by adjusting one or more drift parameters.** For instance, increase the "Mean Shift (Feature 0)" to `1.0` or the "Concept Drift Factor" to `0.2`.
3.  **Set "Number of steps to simulate (per click)" to `10`.**
4.  **Click the "Run Simulation Steps" button.** Repeat this multiple times to accumulate more historical data and observe trends.

Each click of "Run Simulation Steps" triggers the `run_monitoring_step` function, which:
1.  Generates a new batch of synthetic data with the configured drift parameters.
2.  Appends this batch to the `monitoring_data_X` and `monitoring_data_y` in `st.session_state`.
3.  Evaluates the current Champion model on this new batch to calculate performance metrics.
4.  Calculates data drift metrics (K-S and JSD) for `feature_0` against the baseline.
5.  Checks if any metrics cross predefined alert thresholds.
6.  Updates the historical data stored in `st.session_state`.

```python
# Simplified run_monitoring_step function (from app.py)
def run_monitoring_step(champion_model, historical_data_X, historical_data_y, baseline_X, baseline_y, time_step, drift_params, alert_thresholds):
    current_batch_X, current_batch_y = generate_synthetic_classification_data(
        # ... params with drift_params ...
    )
    # ... concatenate historical data ...
    y_pred_current = predict_with_model(champion_model, current_batch_X)
    performance_metrics = calculate_classification_metrics(current_batch_y, y_pred_current)

    ks_statistic_f0, _ = calculate_data_drift_ks(baseline_X[:, 0], current_batch_X[:, 0])
    jsd_statistic_f0 = calculate_data_drift_jsd(baseline_X[:, 0], current_batch_X[:, 0])

    # ... check alerts and return results ...
```

### Plotting Model Performance Over Time

The dashboard visually tracks the Champion model's performance:

*   **Accuracy:** $A = \frac{TP + TN}{TP + TN + FP + FN}$ (Total correct predictions out of all predictions)
*   **Precision:** $P = \frac{TP}{TP + FP}$ (Proportion of positive identifications that were actually correct)
*   **Recall:** $R = \frac{TP}{TP + FN}$ (Proportion of actual positives that were identified correctly)
*   **F1-Score:** $F1 = 2 \times \frac{P \times R}{P + R}$ (Harmonic mean of precision and recall)

Observe how these metrics change as you introduce drift. For example, concept drift or performance degradation will likely cause a drop in accuracy. The red dotted line indicates the `Min. Accuracy Threshold` for alerts.

### Understanding and Detecting Data Drift

Data drift, also known as **covariate shift**, occurs when the statistical properties of the input features $P(X)$ change over time. This can lead to a decline in model performance even if the underlying relationship $P(Y|X)$ remains constant. The dashboard uses two common statistical tests to detect data drift on `feature_0`:

*   **Kolmogorov-Smirnov (K-S) Statistic:**
    This non-parametric test quantifies the maximum difference between the empirical cumulative distribution functions (CDFs) of two samples. A higher K-S statistic indicates a greater difference between the distributions.
    $$D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)|$$
    Where $F_{1,n}(x)$ and $F_{2,m}(x)$ are the empirical CDFs of the two samples (baseline vs. current batch).

*   **Jensen-Shannon Divergence (JSD):**
    JSD is a symmetric and smoothed version of the Kullback-Leibler divergence, measuring the similarity between two probability distributions. It's often preferred for drift detection because it's always non-negative, symmetric, and bounded (values between 0 and 1, or 0 and $\log 2$ depending on the base of the logarithm). In our case, $JSD^2$ is used, so the range is 0 to 1.
    $$JSD(P||Q) = \frac{1}{2} D_{KL}(P||M) + \frac{1}{2} D_{KL}(Q||M)$$
    Where $M = \frac{1}{2}(P+Q)$ and $D_{KL}(P||Q) = \sum_{i} P(i) \log \left( \frac{P(i)}{Q(i)} \right)$.

The dashboard plots these drift metrics over time for `feature_0`. The orange and purple dotted lines represent the alert thresholds for K-S Statistic and JSD, respectively. You will observe these metrics increase when you apply "Mean Shift" or "Std Dev Factor" in the simulation parameters.

### Configuring Alert Thresholds and Triggering Alerts

The "Configure Alert Thresholds" expander allows you to set the sensitivity of the monitoring system.

*   **Min. Accuracy Threshold**: If the model's accuracy drops below this value, an alert is triggered.
*   **Max. K-S Stat Threshold**: If the K-S statistic for `feature_0` exceeds this value, a data drift alert is triggered.
*   **Max. JSD Threshold**: If the JSD for `feature_0` exceeds this value, another data drift alert is triggered.

<aside class="negative">
<b>Warning:</b> Misconfiguring thresholds can lead to either excessive "alert fatigue" (too many false positives) or missed critical issues (too many false negatives). Finding the right balance requires domain expertise and historical data analysis.
</aside>

After running several simulation steps, check the "Active Alerts" section. If any metrics crossed their thresholds, you will see warnings indicating which alerts were triggered at which time steps.

### Feature Distribution: Baseline vs. Current

At the bottom of the monitoring section, a histogram visualizes the distribution of `feature_0` from the initial baseline against the `monitoring_data_X` accumulated during the simulation. This visual comparison is a powerful way to intuitively grasp the extent and nature of data drift. When you apply `mean_shift` or `std_factor`, you will clearly see the current distribution moving or stretching away from the baseline.

## 5. Implementing Intervention Strategies
Duration: 0:15

When alerts are triggered, or model health metrics show significant deterioration, intervention is necessary. This section demonstrates common strategies for addressing model degradation and drift.

Make sure you have run several simulation steps in Step 4, ideally triggering some alerts.

### Simulating Retraining a "Challenger" Model

A common intervention is to retrain the model on more recent, potentially drifted data. This new model is called a "Challenger" as it will challenge the current "Champion" model.

```python
# From application_pages/dashboard.py main function
if st.button("Simulate Retraining Challenger Model"):
    if not st.session_state.monitoring_data_X.empty:
        X_retrain = st.session_state.monitoring_data_X.values
        y_retrain = st.session_state.monitoring_data_y
        challenger_random_state = st.session_state.random_state + 1 if 'random_state' in st.session_state else 43
        st.session_state.challenger_model = train_logistic_regression_model(X_retrain, y_retrain, random_state=challenger_random_state)
        st.success("Challenger Model retrained successfully on recent monitoring data!")
    else:
        st.warning("No monitoring data accumulated yet. Run some simulation steps first.")
```

1.  **Click the "Simulate Retraining Challenger Model" button.**
    The Challenger model is trained on all the accumulated `monitoring_data_X` and `monitoring_data_y` from the simulation steps you've run. This simulates retraining on the most recent, and potentially drifted, data.

### Champion-Challenger Comparison and Model Promotion

After training a Challenger, the next logical step is to compare its performance against the current Champion model. The goal is to determine if the Challenger is better equipped to handle the current data distribution and concept.

```python
# From application_pages/dashboard.py main function
if st.button("Compare Champion vs. Challenger"):
    st.info("Comparing Champion and Challenger models on the latest monitoring data batch...")

    # Evaluate Champion on current monitoring data
    y_pred_champion = predict_with_model(st.session_state.champion_model, st.session_state.monitoring_data_X.values)
    champion_metrics_current = calculate_classification_metrics(st.session_state.monitoring_data_y, y_pred_champion)
    champion_ks_f0, _ = calculate_data_drift_ks(st.session_state.baseline_X[:,0], st.session_state.monitoring_data_X['feature_0'])
    champion_jsd_f0 = calculate_data_drift_jsd(st.session_state.baseline_X[:,0], st.session_state.monitoring_data_X['feature_0'])
    champion_drift_metrics_current = {'K-S Stat (F0)': champion_ks_f0, 'JSD (F0)': champion_jsd_f0}

    # Evaluate Challenger on current monitoring data
    y_pred_challenger = predict_with_model(st.session_state.challenger_model, st.session_state.monitoring_data_X.values)
    challenger_metrics_current = calculate_classification_metrics(st.session_state.monitoring_data_y, y_pred_challenger)
    challenger_ks_f0, _ = calculate_data_drift_ks(st.session_state.baseline_X[:,0], st.session_state.monitoring_data_X['feature_0'])
    challenger_jsd_f0 = calculate_data_drift_jsd(st.session_state.baseline_X[:,0], st.session_state.monitoring_data_X['feature_0'])
    challenger_drift_metrics_current = {'K-S Stat (F0)': challenger_ks_f0, 'JSD (F0)': challenger_jsd_f0}

    compare_champion_challenger(champion_metrics_current, challenger_metrics_current, champion_drift_metrics_current, challenger_drift_metrics_current)

    if challenger_metrics_current['accuracy'] > champion_metrics_current['accuracy']:
        st.success("Challenger model shows better accuracy! Consider promoting it.")
    else:
        st.info("Champion model maintains better or similar accuracy.")
```

1.  **Click the "Compare Champion vs. Challenger" button.**
    The dashboard will display two bar charts: one comparing performance metrics (accuracy, precision, recall, f1-score) and another comparing drift metrics (K-S, JSD) of both models, evaluated on the latest monitoring data.

    Based on this comparison, you can decide whether the Challenger model is superior. If it shows significantly better performance, especially if the Champion's performance has degraded due to drift, you might promote the Challenger.

2.  **Click the "Promote Challenger to Champion" button.**
    If you decide to promote, the Challenger model replaces the current Champion. The monitoring history is reset, and the new Champion starts fresh, establishing new baseline performance based on the *current* data, ready for further continuous monitoring.

    ```python
    # From application_pages/dashboard.py main function
    if st.button("Promote Challenger to Champion"):
        if st.session_state.challenger_model is not None:
            st.session_state.champion_model = st.session_state.challenger_model
            st.session_state.challenger_model = None # Reset Challenger

            # Re-evaluate baseline metrics for the new Champion on current data
            y_pred_new_champion_on_current_data = predict_with_model(st.session_state.champion_model, st.session_state.monitoring_data_X.values)
            st.session_state.baseline_metrics = calculate_classification_metrics(st.session_state.monitoring_data_y, y_pred_new_champion_on_current_data)

            # Reset monitoring history to start fresh with the new Champion
            st.session_state.historical_accuracy = [st.session_state.baseline_metrics['accuracy']]
            # ... reset other historical metrics ...
            st.session_state.current_t_step = 0
            st.session_state.monitoring_data_X = pd.DataFrame()
            st.session_state.monitoring_data_y = np.array([])
            st.success("Challenger model promoted to Champion! Monitoring restarted with new Champion.")
    ```

### Human-in-the-Loop and Governance (SR 11-7)

Automated monitoring and intervention are powerful, but human oversight remains critical. SR 11-7 emphasizes the need for qualitative assessment and expert judgment. A "manual review" step acknowledges that complex drift scenarios or unexpected model behaviors may require human investigation, beyond what automated metrics can capture.

1.  **Click the "Trigger Manual Review" button.**
    This simulates initiating a manual review process. It logs the current time step and any active alert details, indicating that a deeper, human-led investigation is needed.

    ```python
    # From application_pages/dashboard.py main function
    if st.button("Trigger Manual Review"):
        alert_details = {}
        if st.session_state.historical_alerts:
            last_alerts = st.session_state.historical_alerts[-1]['alerts']
            for k, v in last_alerts.items():
                alert_details[k] = v
        else:
            alert_details['status'] = "No specific alerts, triggered manually."
        log_manual_review(st.session_state.current_t_step, alert_details)
        st.info("Manual review triggered. A deeper investigation into model and data health is initiated.")
    ```
    This function `log_manual_review` simply prints an `st.info` message to the dashboard, simulating a logging event.

## 6. Conclusion: Continuous Monitoring for Trustworthy AI
Duration: 0:05

You have successfully navigated through a comprehensive simulation of AI model health monitoring. This interactive dashboard provides a hands-on experience in understanding and managing the lifecycle of AI models in a production environment, with a strong emphasis on Model Risk Management (MRM) principles and regulatory alignment (e.g., SR 11-7).

### Key Takeaways

*   **Proactive Detection**: Continuous monitoring allows for the early detection of model degradation, data drift, and concept drift, enabling timely interventions.
*   **Metric-Driven Decisions**: Performance metrics (Accuracy, Precision, Recall, F1-Score) and drift metrics (K-S Statistic, Jensen-Shannon Divergence) provide quantitative evidence for model health.
*   **Adaptive Strategies**: Retraining models on new data and using Champion-Challenger frameworks are effective ways to adapt to evolving data landscapes.
*   **Human Oversight**: Despite automation, human-in-the-loop processes and manual reviews are indispensable for complex problem-solving and ensuring governance.
*   **Regulatory Compliance**: Implementing robust monitoring and intervention strategies is crucial for meeting regulatory expectations for trustworthy and reliable AI systems.

This simulator illustrates that the deployment of an AI model is not the end of its development cycle, but rather the beginning of a continuous process of validation, monitoring, and adaptation. By embracing these practices, organizations can build and maintain confidence in their AI systems, ensuring they remain valuable and compliant assets.

### Further Exploration

*   **Experiment with different drift scenarios**: Observe how specific drift types impact various metrics.
*   **Implement more sophisticated models**: Replace `LogisticRegression` with more complex models (e.g., RandomForest, XGBoost) and see how their robustness to drift changes.
*   **Expand drift detection**: Implement drift detection for multiple features or multivariate drift detection.
*   **Add explainability**: Integrate tools like SHAP or LIME to understand *why* a model's predictions are changing as drift occurs.
*   **Connect to real data**: Adapt the data generation step to load and monitor real-world datasets for actual deployment scenarios.

Thank you for completing this codelab! We hope it has provided you with valuable insights into the critical field of AI Model Health Monitoring and MRM.
