
## Streamlit Application Requirements Specification: AI Model Health Monitor for Risk Managers

### 1. Application Overview

The Streamlit application will guide a **Risk Manager** (our persona) through a simulated scenario of overseeing a critical AI model in a production environment. The narrative focuses on the dynamic challenges of maintaining AI model health, emphasizing the practical application of Model Risk Management (MRM) principles, particularly those adapted from SR 11-7.

**Story-Driven Workflow:**
The user will begin by acknowledging their role and the regulatory context of AI MRM. They will then:
1.  **Establish a Baseline:** Understand the model's initial training data and expected performance.
2.  **Deploy a Champion Model:** Simulate putting a model into production.
3.  **Initiate Monitoring:** Observe the model's health metrics over simulated time steps.
4.  **Encounter Degradation & Drift:** Witness the model's performance decline due to various factors (performance degradation, data drift, concept drift).
5.  **Proactively Manage Risk:** Configure alert thresholds to flag issues promptly.
6.  **Intervene:** Trigger the retraining of a new "Challenger" model.
7.  **Evaluate & Promote:** Compare the Challenger against the Champion and decide on deployment.
8.  **Ensure Governance:** Simulate human oversight for critical decisions.

**Real-World Problem:** Financial institutions deploying AI models face the challenge of continuously validating their reliability, fairness, and compliance with regulatory frameworks like SR 11-7. AI models are not static; they degrade, encounter new data, and their underlying relationships can change, leading to significant financial and reputational risks if not proactively managed. The persona needs tools to detect, diagnose, and address these issues effectively.

**How the Streamlit App Helps the Persona:**
The application provides a hands-on, interactive experience where the Risk Manager can:
-   **Visualize Model Behavior:** See real-time (simulated) changes in performance and data characteristics.
-   **Experiment with Controls:** Adjust alert thresholds and trigger interventions to understand their impact directly.
-   **Gain Insights:** Learn to distinguish between different types of model degradation and data/concept drift.
-   **Reinforce MRM Principles:** Understand the practical implementation of continuous monitoring, champion-challenger frameworks, and human-in-the-loop processes as essential components of robust AI MRM.

**Learning Goals (Applied Skills):**
By interacting with this application, the Risk Manager will gain applied skills to:
-   **Diagnose Model Health Issues:** Interpret time-series plots and distribution comparisons to identify and differentiate between performance degradation, data drift (covariate shift), and concept drift.
-   **Configure Adaptive Monitoring:** Set and adjust alert thresholds for key performance and drift metrics based on a dynamic risk appetite, understanding the trade-offs.
-   **Evaluate Intervention Strategies:** Apply the Champion-Challenger framework to assess the efficacy of model retraining as a corrective action.
-   **Implement Governance Protocols:** Understand the role and necessity of human-in-the-loop decisions and model promotion within an AI MRM lifecycle.
-   **Connect Theory to Practice:** Translate SR 11-7 principles (e.g., continuous validation, independent review) into practical, interactive AI model management actions.

---

### 2. User Interface Requirements

The UI will be structured as a multi-page Streamlit application, guiding the Risk Manager sequentially through the narrative. Each "page" will correspond to a step in the story, ensuring a coherent flow while allowing users to revisit previous steps (though progression through the narrative is sequential).

#### Layout & Navigation Structure
-   **Overall Layout:** A persistent sidebar for navigation and high-level status, with the main content area for the current story step.
-   **Sidebar (`st.sidebar`):**
    -   **Application Title:** "AI Model Health Dashboard"
    -   **Persona Context:** "Role: Risk Manager"
    -   **Current Model Status:** Displays the current Champion model name (e.g., "Logistic Regression"), and its general health status (e.g., "Stable," "Degradation Detected," "Drift Warning"). This updates dynamically.
    -   **Navigation Menu:** A list of `st.sidebar.button`s or `st.sidebar.radio` options, representing the story steps. Buttons for future steps will be disabled until the current step is completed.
        1.  Welcome & Your Role
        2.  Establishing the Baseline
        3.  Training the Champion
        4.  Reviewing Baseline Performance
        5.  Initial Stable Monitoring
        6.  Detecting Performance Degradation
        7.  Detecting Data Drift
        8.  Detecting Concept Drift
        9.  Configuring Alert Thresholds
        10. Retraining a Challenger
        11. Champion-Challenger Comparison
        12. Human-in-the-Loop & Governance
        13. Conclusion
-   **Main Content Area:** (`st.container` or direct layout calls) This area will house the narrative text, input widgets, visualizations, and dynamic feedback for the current story step.

#### Input Widgets and Controls

1.  **Page: "Configuring Alert Thresholds"**
    *   **Purpose in Story:** The Risk Manager defines the acceptable boundaries for model performance and data stability, reflecting organizational risk appetite.
    *   **Real-world Action:** Setting risk thresholds for key performance indicators (KPIs) and data integrity metrics.
    *   **Widget:** `st.slider`
        *   **Parameter:** `Min. Accuracy Threshold`
        *   **Input Type:** Float
        *   **Range:** $0.0$ to $1.0$
        *   **Default:** $0.85$
        *   **Purpose:** Sets the minimum acceptable accuracy for the model. Crossing below this triggers an alert.
        *   **Tooltip:** "Adjust this slider to set the minimum acceptable accuracy score for the model. A lower value indicates higher risk tolerance."
    *   **Widget:** `st.slider`
        *   **Parameter:** `Max. K-S Statistic Threshold (Feature 0)`
        *   **Input Type:** Float
        *   **Range:** $0.0$ to $1.0$
        *   **Default:** $0.15$
        *   **Purpose:** Sets the maximum acceptable Kolmogorov-Smirnov (K-S) statistic for comparing baseline and current feature distributions. Exceeding this triggers an alert for data drift.
        *   **Tooltip:** "Set the maximum tolerable difference in feature distributions as measured by the K-S test. Higher values mean more drift."
    *   **Widget:** `st.slider`
        *   **Parameter:** `Max. JSD Threshold (Feature 0)`
        *   **Input Type:** Float
        *   **Range:** $0.0$ to $1.0$
        *   **Default:** $0.15$
        *   **Purpose:** Sets the maximum acceptable Jensen-Shannon Divergence (JSD) for comparing baseline and current feature distributions. Exceeding this triggers an alert for data drift.
        *   **Tooltip:** "Set the maximum tolerable dissimilarity between feature distributions using Jensen-Shannon Divergence. Higher values mean greater divergence."

#### Visualization Components

All visualizations will use `plotly.graph_objects` via `st.plotly_chart()`. Each plot will have clear titles, axis labels, and legends.

1.  **Page: "Initial Stable Monitoring", "Detecting Performance Degradation", "Detecting Data Drift", "Detecting Concept Drift", "Configuring Alert Thresholds"**
    *   **Chart Type:** Line Plot with Markers (`go.Scatter`)
    *   **Data:** Historical Accuracy, Precision, Recall, F1-Score over time.
    *   **Format:**
        *   X-axis: 'Time Step' (integer from 1 to current simulation step)
        *   Y-axis: 'Score' (float from $0.0$ to $1.0$)
        *   **Expected Output:** Multiple lines representing each metric. A horizontal red dashed line indicating the `Min. Accuracy Threshold`. Points on the accuracy line will be highlighted or annotated when an alert is triggered.
    *   **Purpose in Story:** Allows the Risk Manager to track the model's performance evolution, immediately identifying when it degrades or crosses a critical threshold.
    *   **Tie to Insight:** "Is our model still performing as expected? Have we hit a regulatory limit?"

2.  **Page: "Initial Stable Monitoring", "Detecting Data Drift", "Detecting Concept Drift", "Configuring Alert Thresholds"**
    *   **Chart Type:** Line Plot with Markers (`go.Scatter`)
    *   **Data:** Historical K-S Statistic and JSD for `feature_0` over time.
    *   **Format:**
        *   X-axis: 'Time Step'
        *   Y-axis 1 (left): 'K-S Statistic'
        *   Y-axis 2 (right, overlaying): 'JSD'
        *   **Expected Output:** Two lines showing the trend of drift metrics. Horizontal orange and purple dashed lines indicating `Max. K-S Statistic Threshold` and `Max. JSD Threshold`, respectively. Points on the lines will be highlighted or annotated when an alert is triggered.
    *   **Purpose in Story:** Helps the Risk Manager detect and understand shifts in input data distributions over time, correlating these changes with potential performance drops.
    *   **Tie to Insight:** "Is the data fed to our model still similar to what it was trained on? Is this drift causing performance issues?"

3.  **Page: "Detecting Data Drift"**
    *   **Chart Type:** Overlaid Histograms (`go.Histogram`)
    *   **Data:** Distribution of `feature_0` from the `baseline_df` vs. `current_monitoring_data_X` (recent batch).
    *   **Format:**
        *   X-axis: 'Feature 0 Value'
        *   Y-axis: 'Density'
        *   **Expected Output:** Two overlaid histograms (e.g., blue for baseline, red for current), normalized to probability density, demonstrating visual differences in distributions.
    *   **Purpose in Story:** Provides a direct visual comparison of feature distributions, reinforcing the statistical drift metrics.
    *   **Tie to Insight:** "Visually, how much has our data changed? Is the magnitude of this shift significant enough to warrant investigation?"

4.  **Page: "Champion-Challenger Comparison"**
    *   **Chart Type:** Grouped Bar Chart (`go.Bar`)
    *   **Data:** Comparison of Champion vs. Challenger model performance metrics (Accuracy, Precision, Recall, F1-Score) on the *same recent monitoring data*.
    *   **Format:**
        *   X-axis: 'Metric Type'
        *   Y-axis: 'Score'
        *   **Expected Output:** Two bars per metric (blue for Champion, red for Challenger) grouped together, allowing for easy side-by-side comparison.
    *   **Purpose in Story:** Empowers the Risk Manager to make an data-driven decision about which model is superior for production deployment.
    *   **Tie to Insight:** "Does our new Challenger model truly outperform the current Champion, especially on the problematic recent data?"

5.  **Page: "Champion-Challenger Comparison"**
    *   **Chart Type:** Grouped Bar Chart (`go.Bar`)
    *   **Data:** Comparison of Champion vs. Challenger model drift metrics (K-S Statistic, JSD for Feature 0) on the *same recent monitoring data* (compared against the original baseline).
    *   **Format:**
        *   X-axis: 'Drift Metric'
        *   Y-axis: 'Drift Statistic'
        *   **Expected Output:** Two bars per drift metric, grouped, showing which model's predictions or internal states are less impacted by the observed data shift. (Note: The drift metrics are on the *data* itself, not the model output, but this visualization will include performance metrics for a comprehensive view, as per notebook's `champion_drift_metrics` structure which includes performance. The primary drift metrics K-S and JSD will implicitly reflect the impact if the Challenger was trained on the drifted data, improving its 'fit' to the current data distribution).
    *   **Purpose in Story:** To understand if the new model inherently handles or is more robust to the observed drift.
    *   **Tie to Insight:** "Has retraining effectively 're-aligned' the model to the current data environment, reducing the apparent drift?"

#### Interactive Elements & Feedback Mechanisms

1.  **Navigation Buttons (e.g., `st.button` for "Next Step", "Start Monitoring"):**
    *   **Purpose:** Guide the user sequentially through the story, progressing from one concept to the next.
    *   **Feedback:** Update `st.session_state.current_page` and re-render the main content area with the next page's elements. Sidebar navigation buttons will dynamically enable/disable based on `current_page`.

2.  **Sliders (on "Configuring Alert Thresholds" page):**
    *   **Purpose:** Allow the Risk Manager to dynamically adjust alert thresholds.
    *   **Feedback:** `on_change` callback function (`update_dashboard_streamlit`) will:
        *   Update `st.session_state.current_alert_thresholds`.
        *   Re-run the entire historical simulation logic (but not data generation) using the updated thresholds.
        *   Re-render the performance and drift plots, showing new horizontal threshold lines and highlighting any points that now trigger alerts.
        *   Update a dedicated `st.text_area` (`"Alert Log"`) to display which alerts were triggered at which time steps, based on the *new* thresholds.

3.  **`Simulate Retraining Challenger Model` Button (on "Retraining a Challenger" page):**
    *   **Purpose:** Initiate the process of training a new model on combined baseline and recent production data.
    *   **Feedback:**
        *   Displays "Simulating model retraining..." immediately below the button.
        *   After computation, displays "Challenger Model trained successfully."
        *   Presents a table/metrics of the Challenger's initial performance on recent monitoring data.
        *   Enables the "Compare Champion vs. Challenger" button.

4.  **`Compare Champion vs. Challenger` Button (on "Champion-Challenger Comparison" page):**
    *   **Purpose:** Trigger the comparative analysis of the current production model and the newly trained challenger.
    *   **Feedback:**
        *   Displays the two grouped bar charts (`fig_perf_comp`, `fig_drift_comp`) directly below the button.
        *   Provides textual analysis guidance ("Observe how the Challenger model performs...").
        *   Enables the "Promote Challenger to Champion" button.

5.  **`Promote Challenger to Champion` Button (on "Champion-Challenger Comparison" page):**
    *   **Purpose:** The Risk Manager decides to replace the current production model with the superior Challenger.
    *   **Feedback:**
        *   Displays "Challenger Model successfully promoted to become the new Champion!"
        *   Updates `st.session_state.champion_model` to the challenger model.
        *   Updates the sidebar's "Current Model Status."
        *   Resets the historical monitoring data (or a portion of it) to reflect the new Champion's fresh start, or continues monitoring with the new champion (for simplicity, we will assume a reset for a new monitoring cycle if time permits, otherwise a simple confirmation is sufficient).

6.  **`Trigger Manual Review` Button (on "Human-in-the-Loop & Governance" page, and potentially accessible via sidebar during alerts):**
    *   **Purpose:** Simulate a human intervention to investigate an alert or model issue that requires qualitative assessment or policy decision.
    *   **Feedback:**
        *   Displays "Manual review triggered. A risk manager would now investigate and decide on appropriate actions."
        *   Logs an entry to a persistent `st.text_area` or internal log (`st.session_state.manual_review_log`) detailing the current time step and any active alerts.

---

### 3. Additional Requirements

#### Annotations & Tooltips
-   **Plots:**
    -   All horizontal threshold lines (`st.plotly_chart.add_hline`) will include `annotation_text` explaining what the line represents (e.g., "Min. Accuracy Threshold").
    -   Hover-over tooltips on all time-series plots will show the exact metric value and time step.
    -   For alerts triggered on plots, specific data points will be visually distinct (e.g., larger marker size, different color) and their hover text will include the alert details.
-   **Metrics:**
    -   `st.metric` displays for Accuracy, Precision, Recall, F1-Score on the "Reviewing Baseline Performance" page will have tooltips:
        -   **Accuracy:** "The proportion of correctly classified instances. A drop below the set threshold indicates a performance risk."
        -   **Precision:** "The proportion of positive identifications that were actually correct. Crucial when false positives are costly."
        -   **Recall:** "The proportion of actual positives that were identified correctly. Important when false negatives are costly."
        -   **F1-Score:** "The harmonic mean of Precision and Recall, offering a balanced view of performance."
    -   `st.metric` displays for K-S Statistic and JSD (if shown as standalone metrics) will have tooltips:
        -   **K-S Statistic:** "Measures the maximum difference between the cumulative distributions of two data samples. High values suggest data drift."
        -   **JSD:** "A symmetric measure of similarity between two probability distributions. Higher values indicate greater dissimilarity or data drift."
-   **Widgets:**
    -   All `st.slider` widgets for threshold adjustment will include `st.help()` or descriptive `st.markdown` nearby explaining their purpose in the context of risk management (as described in "Input Widgets").
    -   All `st.button` widgets will have a clear description.

#### State Management Requirements
-   **`st.session_state` Usage:** All critical application data and user inputs will be stored in `st.session_state`.
    -   `st.session_state.current_page`: Tracks the user's current position in the story workflow.
    -   `st.session_state.baseline_X`, `st.session_state.baseline_y`, `st.session_state.baseline_df`: Stores the initial synthetic dataset.
    -   `st.session_state.champion_model`: The currently active production model object.
    -   `st.session_state.challenger_model`: The challenger model object (None if not yet trained).
    -   `st.session_state.monitoring_data_X`, `st.session_state.monitoring_data_y`: Accumulates all `current_batch_X` and `current_batch_y` generated during monitoring steps.
    -   `st.session_state.historical_metrics`: A list of dictionaries, where each dictionary stores all calculated performance and drift metrics, the `drift_parameters` used for that step, and any alerts triggered for a specific time step. This is essential for dynamic threshold re-evaluation.
    -   `st.session_state.current_alert_thresholds`: A dictionary containing the currently set accuracy, K-S, and JSD thresholds.
    -   `st.session_state.max_time_step_reached`: Tracks the highest `t_step` simulated, for re-running the historical simulation with new thresholds.
    -   `st.session_state.simulation_logs`: A dictionary of lists of strings, storing logs for different simulation phases, to be displayed in `st.text_area`.
    -   `st.session_state.manual_review_log`: A list of strings logging manual review events.
-   **Persistence:** All `st.session_state` variables will persist across reruns and interactions within the application, ensuring the user does not lose progress as they move through the scenario or adjust parameters.
-   **Initialization:** `st.session_state` variables will be initialized on the first run of the application.
-   **Serialization:** Model objects (`champion_model`, `challenger_model`) should be pickleable or managed appropriately for `st.session_state`. `sklearn` models are generally pickleable.

---

### 4. Notebook Content and Code Requirements

All relevant code stubs and markdown explanations from the Jupyter Notebook will be translated into Streamlit components and functions.

#### 4.1. Core Functions (Defined globally in `streamlit_app.py`)

-   **`generate_synthetic_classification_data(...)`**:
    *   **Description:** Creates synthetic datasets with optional drift factors.
    *   **Integration:** Used for initial `baseline_df` generation and for simulating new `current_batch_X`, `current_batch_y` at each monitoring step.
    *   **Streamlit Page:** "Establishing the Baseline", and internally during monitoring steps on "Initial Stable Monitoring" through "Configuring Alert Thresholds."
-   **`train_logistic_regression_model(X_train, y_train, random_state=42)`**:
    *   **Description:** Trains a `LogisticRegression` model.
    *   **Integration:** Used to train the initial `champion_model` and subsequently the `challenger_model`.
    *   **Streamlit Page:** "Training the Champion", "Retraining a Challenger."
-   **`predict_with_model(model, X_data)`**:
    *   **Description:** Generates predictions from a given model.
    *   **Integration:** Called to get predictions for performance metric calculations for both champion and challenger models.
    *   **Streamlit Page:** Internally within `calculate_classification_metrics` and `run_monitoring_step`.
-   **`calculate_classification_metrics(y_true, y_pred)`**:
    *   **Description:** Computes Accuracy, Precision, Recall, F1-Score.
    *   **Integration:** Used to evaluate baseline, champion, and challenger model performance.
    *   **Streamlit Page:** "Reviewing Baseline Performance", internally during monitoring steps, "Retraining a Challenger", "Champion-Challenger Comparison."
-   **`calculate_data_drift_ks(baseline_feature_data, current_feature_data)`**:
    *   **Description:** Calculates the Kolmogorov-Smirnov (K-S) statistic.
    *   **Integration:** Used to detect data drift for selected features.
    *   **Streamlit Page:** Internally during monitoring steps, "Detecting Data Drift."
-   **`calculate_data_drift_jsd(baseline_feature_data, current_feature_data, num_bins=50)`**:
    *   **Description:** Calculates Jensen-Shannon Divergence (JSD).
    *   **Integration:** Used to detect data drift for selected features.
    *   **Streamlit Page:** Internally during monitoring steps, "Detecting Data Drift."
-   **`run_monitoring_step(champion_model, historical_data_X, historical_data_y, baseline_X, baseline_y, time_step, drift_params, alert_thresholds)`**:
    *   **Description:** Simulates one batch of production data, calculates metrics, checks for alerts.
    *   **Integration:** This function will be called iteratively or within a loop to simulate the continuous monitoring process. It will be central to updating the `historical_metrics` and `monitoring_data_X/y` in `st.session_state`.
    *   **Streamlit Page:** "Initial Stable Monitoring" through "Configuring Alert Thresholds."
-   **`retrain_challenger_model(X_train_challenger, y_train_challenger, random_state=42)`**:
    *   **Description:** Trains a new Challenger model.
    *   **Integration:** Triggered by a `st.button` on the "Retraining a Challenger" page.
    *   **Streamlit Page:** "Retraining a Challenger."
-   **`compare_champion_challenger(champion_metrics, challenger_metrics, champion_drift_metrics, challenger_drift_metrics)`**:
    *   **Description:** Generates plotly bar charts for model comparison.
    *   **Integration:** Called by a `st.button` on the "Champion-Challenger Comparison" page.
    *   **Streamlit Page:** "Champion-Challenger Comparison."
-   **`log_manual_review(time_step, alert_details)`**:
    *   **Description:** Records a manual review event.
    *   **Integration:** Triggered by a `st.button` on the "Human-in-the-Loop & Governance" page.
    *   **Streamlit Page:** "Human-in-the-Loop & Governance."

#### 4.2. Streamlit Page-Specific Implementations (Mapping Notebook Cells)

**Page 1: Welcome & Introduction to AI MRM**
-   **Markdown:** `st.markdown()` for content from notebook cells "AI Model Health Dashboard Simulator Notebook" and "Core Concepts: SR 11-7 and AI Model Risk Management".
-   **Interaction:** `st.button("Start My Monitoring Journey")` to advance page.

**Page 2: Establishing the Baseline**
-   **Markdown:** `st.markdown()` for "Setting Up the Environment and Generating Baseline Data".
-   **Code (Initial Run):**
    ```python
    # Initial generation, run once if not in session state
    if 'baseline_X' not in st.session_state:
        st.session_state.baseline_X, st.session_state.baseline_y = generate_synthetic_classification_data(
            num_samples=1000, num_features=5, n_informative=3, n_redundant=0,
            n_clusters_per_class=1, random_state=42
        )
        st.session_state.baseline_df = pd.DataFrame(st.session_state.baseline_X, columns=[f'feature_{i}' for i in range(st.session_state.baseline_X.shape[1])])
        st.session_state.baseline_df['target'] = st.session_state.baseline_y
    ```
-   **Output:** `st.dataframe(st.session_state.baseline_df.head())`, `st.write(f"Baseline Data Shape: {st.session_state.baseline_df.shape}")`, `st.write(f"Baseline Target Distribution:\n{st.session_state.baseline_df['target'].value_counts(normalize=True)}")`.
-   **Markdown:** `st.markdown()` for "The output above shows a glimpse...".

**Page 3: Training the Champion**
-   **Markdown:** `st.markdown()` for "Training the Initial "Champion" Model".
-   **Code (Initial Run):**
    ```python
    if 'champion_model' not in st.session_state:
        X_train_baseline, X_test_baseline, y_train_baseline, y_test_baseline = train_test_split(
            st.session_state.baseline_X, st.session_state.baseline_y, test_size=0.3, random_state=42
        )
        st.session_state.champion_model = train_logistic_regression_model(X_train_baseline, y_train_baseline, random_state=42)
        st.session_state.X_test_baseline = X_test_baseline # Store for baseline metrics
        st.session_state.y_test_baseline = y_test_baseline
    ```
-   **Output:** `st.success("Champion Model trained successfully.")`

**Page 4: Reviewing Baseline Performance**
-   **Markdown:** `st.markdown()` for "Defining and Calculating Classification Performance Metrics", including LaTeX formulas (`$$A = \frac{TP + TN}{TP + TN + FP + FN}$$, etc.).
-   **Code (Initial Run):**
    ```python
    if 'baseline_metrics' not in st.session_state:
        y_pred_baseline = predict_with_model(st.session_state.champion_model, st.session_state.X_test_baseline)
        st.session_state.baseline_metrics = calculate_classification_metrics(st.session_state.y_test_baseline, y_pred_baseline)
    ```
-   **Output:** `st.metric` for each baseline metric.
-   **Markdown:** `st.markdown()` for "These baseline metrics represent...".

**Page 5: Initial Stable Monitoring**
-   **Markdown:** `st.markdown()` for "Simulating Production Data Stream and Monitoring Framework" (adjusted).
-   **Code (Simulation Loop - initial 5 stable steps):**
    ```python
    # Initialize monitoring state if not present
    if 'historical_metrics' not in st.session_state:
        st.session_state.historical_metrics = []
        st.session_state.monitoring_data_X = pd.DataFrame(columns=[f'feature_{i}' for i in range(st.session_state.baseline_X.shape[1])])
        st.session_state.monitoring_data_y = np.array([])
        st.session_state.current_alert_thresholds = {'accuracy_min': 0.85, 'ks_max': 0.15, 'jsd_max': 0.15}
        st.session_state.drift_parameters = {'mean_shift': 0, 'std_factor': 1, 'concept_drift_factor': 0, 'performance_degradation_factor': 0}
        st.session_state.max_time_step_reached = 0

    # Simulate 5 stable steps if not already done
    if st.session_state.max_time_step_reached < 5:
        for t_step in range(st.session_state.max_time_step_reached + 1, 6):
            step_results = run_monitoring_step(
                st.session_state.champion_model,
                st.session_state.monitoring_data_X,
                st.session_state.monitoring_data_y,
                st.session_state.baseline_X,
                st.session_state.baseline_y,
                t_step,
                st.session_state.drift_parameters, # Stable drift params
                st.session_state.current_alert_thresholds
            )
            # Store full results for re-evaluation with new thresholds later
            st.session_state.historical_metrics.append({
                'time_step': t_step,
                'metrics': {k: v for k,v in step_results.items() if k not in ['alerts', 'monitoring_data_X', 'monitoring_data_y']},
                'alerts': step_results['alerts'],
                'drift_params_applied': st.session_state.drift_parameters.copy()
            })
            st.session_state.monitoring_data_X = step_results['monitoring_data_X']
            st.session_state.monitoring_data_y = step_results['monitoring_data_y']
            st.session_state.max_time_step_reached = t_step
            # Update log
            # ...
    # Plotting using data from st.session_state.historical_metrics
    # ...
    ```

**Page 6: Detecting Performance Degradation**
-   **Markdown:** `st.markdown()` for "Detecting Performance Degradation Over Time".
-   **Code (Simulation Loop - additional steps with degradation):**
    ```python
    # Simulate steps 6-15 with performance degradation
    if st.session_state.max_time_step_reached < 15:
        for t_step in range(max(6, st.session_state.max_time_step_reached + 1), 16):
            current_drift_params = st.session_state.drift_parameters.copy()
            current_drift_params['performance_degradation_factor'] = min(0.01 * (t_step - 5), 0.15)
            step_results = run_monitoring_step(...) # Call with current_drift_params
            # Append to st.session_state.historical_metrics, update monitoring_data_X/y, max_time_step_reached
            # ...
    # Plotting (fig with degradation)
    # ...
    ```

**Page 7: Understanding and Detecting Data Drift**
-   **Markdown:** `st.markdown()` for "Understanding and Detecting Data Drift (Covariate Shift)", including LaTeX for K-S ($$D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)|$$) and JSD ($$\text{JSD}(P||Q) = \frac{1}{2} D_{\text{KL}}(P||M) + \frac{1}{2} D_{\text{KL}}(Q||M)$$, etc.).
-   **Code (Example Drift Calc):**
    ```python
    # Placeholder for demonstrating drift calculation on example data
    np.random.seed(42)
    sample_a = np.random.normal(loc=0, scale=1, size=100)
    sample_b_no_drift = np.random.normal(loc=0, scale=1, size=100)
    sample_c_with_drift = np.random.normal(loc=1, scale=1.2, size=100)
    ks_no_drift, p_no_drift = calculate_data_drift_ks(sample_a, sample_b_no_drift)
    jsd_no_drift = calculate_data_drift_jsd(sample_a, sample_b_no_drift)
    ks_with_drift, p_with_drift = calculate_data_drift_ks(sample_a, sample_c_with_drift)
    jsd_with_drift = calculate_data_drift_jsd(sample_a, sample_c_with_drift)
    st.write(f"K-S (no drift): {ks_no_drift:.4f}, JSD (no drift): {jsd_no_drift:.4f}")
    st.write(f"K-S (with drift): {ks_with_drift:.4f}, JSD (with drift): {jsd_with_drift:.4f}")
    ```
-   **Markdown:** `st.markdown()` for "The example demonstrates...".
-   **Code (Simulation Loop - additional steps with data drift):**
    ```python
    # Reset historical metrics and data for a new simulation with data drift
    # This might mean clearing and re-running the entire history, or only adding new steps
    # For a story, we can simulate a fresh set of steps 1-5 stable, then 6-15 with data drift.
    # ... logic for simulating/appending ...
    if st.session_state.max_time_step_reached < 30: # Assuming 15 degradation + 15 data drift
        for t_step in range(max(16, st.session_state.max_time_step_reached + 1), 31):
            current_drift_params = st.session_state.drift_parameters.copy()
            current_drift_params['mean_shift'] = min(0.1 * (t_step - 15), 1.0) # Gradually shift mean
            step_results = run_monitoring_step(...) # Call with current_drift_params
            # Append to st.session_state.historical_metrics, update monitoring_data_X/y, max_time_step_reached
            # ...
    # Plotting (K-S, JSD, and distribution comparisons)
    # ...
    ```

**Page 8: Understanding and Detecting Concept Drift**
-   **Markdown:** `st.markdown()` for "Understanding and Detecting Concept Drift".
-   **Code (Simulation Loop - additional steps with concept drift):**
    ```python
    # Reset historical metrics and data for a new simulation with concept drift (similar to data drift)
    # This might mean clearing and re-running the entire history, or only adding new steps
    # ... logic for simulating/appending ...
    if st.session_state.max_time_step_reached < 45: # Assuming 15 prev + 15 this one
        for t_step in range(max(31, st.session_state.max_time_step_reached + 1), 46):
            current_drift_params = st.session_state.drift_parameters.copy()
            current_drift_params['concept_drift_factor'] = min(0.01 * (t_step - 30), 0.1)
            step_results = run_monitoring_step(...) # Call with current_drift_params
            # Append to st.session_state.historical_metrics, update monitoring_data_X/y, max_time_step_reached
            # ...
    # Plotting (accuracy with concept drift, K-S/JSD stable)
    # ...
    ```

**Page 9: Configuring Alert Thresholds**
-   **Markdown:** `st.markdown()` for "Configuring Alert Thresholds and Triggering Alerts".
-   **Widgets:** `st.slider` for `accuracy_min`, `ks_max`, `jsd_max`. Each slider's `on_change` will trigger `update_dashboard_streamlit`.
    ```python
    def update_dashboard_streamlit():
        # Update thresholds in session state
        st.session_state.current_alert_thresholds['accuracy_min'] = st.session_state.accuracy_threshold_slider
        st.session_state.current_alert_thresholds['ks_max'] = st.session_state.ks_threshold_slider
        st.session_state.current_alert_thresholds['jsd_max'] = st.session_state.jsd_threshold_slider

        # Re-evaluate all historical_metrics with new thresholds
        updated_alerts_log = []
        for i, step_data in enumerate(st.session_state.historical_metrics):
            # Recalculate alerts based on new thresholds without regenerating data
            re_evaluated_alerts = {}
            if step_data['metrics']['accuracy'] < st.session_state.current_alert_thresholds['accuracy_min']:
                re_evaluated_alerts['accuracy'] = f"Accuracy threshold crossed: {step_data['metrics']['accuracy']:.4f}"
            if step_data['metrics']['ks_statistic_feature_0'] > st.session_state.current_alert_thresholds['ks_max']:
                re_evaluated_alerts['ks_feature_0'] = f"K-S Stat (Feature 0) threshold crossed: {step_data['metrics']['ks_statistic_feature_0']:.4f}"
            if step_data['metrics']['jsd_statistic_feature_0'] > st.session_state.current_alert_thresholds['jsd_max']:
                re_evaluated_alerts['jsd_feature_0'] = f"JSD (Feature 0) threshold crossed: {step_data['metrics']['jsd_statistic_feature_0']:.4f}"
            
            st.session_state.historical_metrics[i]['alerts'] = re_evaluated_alerts
            if re_evaluated_alerts:
                updated_alerts_log.append(f"ALERT at Time Step {step_data['time_step']}: {re_evaluated_alerts}")
        
        st.session_state.simulation_logs['alerts'] = updated_alerts_log # Update log for display
        
        # Then re-render plots based on updated historical_metrics
        # ... plot generation code ...
    ```
-   **Output:** `st.text_area` for alert log, dynamic `st.plotly_chart` for accuracy and drift metrics with adjustable thresholds.

**Page 10: Retraining a Challenger**
-   **Markdown:** `st.markdown()` for "Simulating Interventions: Retraining a "Challenger" Model".
-   **Widget:** `st.button("Simulate Retraining Challenger Model")`.
-   **Callback for button:**
    ```python
    def on_retrain_clicked():
        st.info("Simulating model retraining...")
        challenger_X_train = pd.concat([pd.DataFrame(st.session_state.baseline_X), st.session_state.monitoring_data_X], ignore_index=True)
        challenger_y_train = np.concatenate([st.session_state.baseline_y, st.session_state.monitoring_data_y])
        st.session_state.challenger_model = retrain_challenger_model(challenger_X_train, challenger_y_train, random_state=42)
        st.success("Challenger Model trained successfully.")
        
        # Evaluate performance for display
        y_pred_challenger_recent = predict_with_model(st.session_state.challenger_model, st.session_state.monitoring_data_X)
        challenger_metrics_recent = calculate_classification_metrics(st.session_state.monitoring_data_y, y_pred_challenger_recent)
        st.write("Challenger Model Performance on Recent Data:")
        # ... display metrics ...
        
        st.session_state.compare_button_enabled = True # Enable next button
    ```

**Page 11: Champion-Challenger Comparison**
-   **Markdown:** `st.markdown()` for "Champion-Challenger Comparison and Model Promotion".
-   **Widget:** `st.button("Compare Champion vs. Challenger")`.
-   **Callback for compare button:**
    ```python
    def on_compare_clicked():
        if st.session_state.challenger_model is None:
            st.warning("Please train the Challenger Model first.")
            return

        # Re-evaluate Champion & Challenger on recent data
        y_pred_champion_recent = predict_with_model(st.session_state.champion_model, st.session_state.monitoring_data_X)
        champion_metrics_recent = calculate_classification_metrics(st.session_state.monitoring_data_y, y_pred_champion_recent)
        y_pred_challenger_recent = predict_with_model(st.session_state.challenger_model, st.session_state.monitoring_data_X)
        challenger_metrics_recent = calculate_classification_metrics(st.session_state.monitoring_data_y, y_pred_challenger_recent)

        # Calculate drift metrics (data drift, not model-specific performance on drift)
        champion_ks_f0, _ = calculate_data_drift_ks(st.session_state.baseline_df['feature_0'], st.session_state.monitoring_data_X['feature_0'])
        champion_jsd_f0 = calculate_data_drift_jsd(st.session_state.baseline_df['feature_0'], st.session_state.monitoring_data_X['feature_0'])
        challenger_ks_f0, _ = calculate_data_drift_ks(st.session_state.baseline_df['feature_0'], st.session_state.monitoring_data_X['feature_0']) # Note: Challenger uses same monitoring data for drift test against original baseline
        challenger_jsd_f0 = calculate_data_drift_jsd(st.session_state.baseline_df['feature_0'], st.session_state.monitoring_data_X['feature_0'])

        # Package metrics for comparison plots as per notebook
        champion_all_metrics = {**champion_metrics_recent, 'K-S Feature 0': champion_ks_f0, 'JSD Feature 0': champion_jsd_f0}
        challenger_all_metrics = {**challenger_metrics_recent, 'K-S Feature 0': challenger_ks_f0, 'JSD Feature 0': challenger_jsd_f0}

        compare_champion_challenger(champion_metrics_recent, challenger_metrics_recent, champion_all_metrics, challenger_all_metrics)
        st.markdown("\nAnalysis: Observe how the Challenger model performs against the Champion model on recent, potentially drifted data. A superior Challenger is a candidate for promotion.")
        st.session_state.promote_button_enabled = True # Enable next button
    ```
-   **Widget:** `st.button("Promote Challenger to Champion")`.
-   **Callback for promote button:**
    ```python
    def on_promote_clicked():
        if st.session_state.challenger_model is not None:
            st.session_state.champion_model = st.session_state.challenger_model
            st.success("Challenger Model successfully promoted to become the new Champion!")
            st.info("Monitoring will now continue with the new Champion model.")
            # Optionally, reset historical_metrics to start fresh monitoring for new Champion
            # or clear challenger_model state
            st.session_state.challenger_model = None
        else:
            st.warning("No Challenger model available to promote.")
    ```

**Page 12: Human-in-the-Loop & Governance**
-   **Markdown:** `st.markdown()` for "Human-in-the-Loop and Governance in AI Monitoring (SR 11-7)".
-   **Widget:** `st.button("Trigger Manual Review")`.
-   **Callback for manual review button:**
    ```python
    def on_manual_review_clicked():
        current_alerts_state = st.session_state.historical_metrics[-1]['alerts'] if st.session_state.historical_metrics else "No recent alerts"
        log_message = f"Manual Review Log: Time Step {st.session_state.max_time_step_reached}, Alert Details: {current_alerts_state}"
        st.session_state.manual_review_log.append(log_message)
        st.info("Manual review triggered. A risk manager would now investigate and decide on appropriate actions.")
        st.text_area("Manual Review Events", value="\n".join(st.session_state.manual_review_log), height=150, disabled=True)
    ```

**Page 13: Conclusion**
-   **Markdown:** `st.markdown()` for "Conclusion: Continuous Monitoring for Trustworthy AI".
-   **Interaction:** `st.button("Restart Simulation")` to clear `st.session_state` and redirect to the first page.

All mathematical content, especially the formulas for Accuracy, Precision, Recall, F1-Score, K-S Test, and JSD, will be rendered using LaTeX with `$$...$$` for display equations and `$...$` for inline as per requirements. No asterisks will be used around mathematical variables.
