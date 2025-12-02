id: 692f416d3bc9bc6a310b84d4_user_guide
summary: AI Design and Deployment Lab 4 User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Navigating AI Model Health: A Risk Manager's Simulation Dashboard

## 1. Introduction to AI Model Risk Management (MRM)
Duration: 00:05:00

Welcome to this codelab! This interactive Streamlit dashboard provides a simulated environment to understand and manage the health of AI models, a crucial aspect for any Risk Manager. In today's rapidly evolving technological landscape, AI models are increasingly integrated into critical business operations. However, like any sophisticated tool, they require diligent oversight to ensure they continue to perform as expected and do not introduce unforeseen risks.

<aside class="positive">
<b>Why is this important?</b> Regulatory guidelines, such as SR 11-7, emphasize the need for continuous monitoring and robust governance frameworks for adaptive AI systems. This dashboard will help you visualize and interact with the core concepts of AI Model Risk Management (MRM) in a practical setting.
</aside>

**Learning Goals:**
*   **Continuous Monitoring**: Learn how to set up and track a model's performance and data characteristics over time, mimicking a real-world production environment.
*   **Drift Detection**: Understand how to identify and distinguish between different types of model degradation:
    *   **Performance Degradation**: When the model's predictive accuracy drops.
    *   **Data Drift ($P(X)$ change)**: When the incoming data's characteristics change compared to the data the model was trained on.
    *   **Concept Drift ($P(Y|X)$ change)**: When the relationship between input features and the target variable changes.
*   **Threshold-based Alerts**: Discover how to configure and react to automated alerts when model health metrics deviate from acceptable norms.
*   **Intervention Strategies**: Simulate critical actions like model retraining, comparing new models ("Challengers") against the existing one ("Champion"), and the process of promoting superior models.
*   **SR 11-7 Principles**: Gain practical insight into ongoing validation and risk control for adaptive AI systems.

This guide will walk you through the dashboard step-by-step, focusing on the *why* and *what* of each functionality from a risk management perspective, rather than the technical implementation details.

## 2. Establishing Your Baseline: The Champion Model
Duration: 00:08:00

Every monitoring process needs a reference point. In AI MRM, this reference is typically your initial, validated "Champion" model and the data it was trained on. This step will guide you through generating synthetic data and training your first Champion model.

1.  **Navigate to the Dashboard:**
    *   Ensure you are on the "Dashboard" page of the application, usually selected by default or via the sidebar.

2.  **Configure Baseline Data Generation:**
    *   Look for the section titled "1. Baseline Setup".
    *   Click on the "Generate Baseline Data & Train Champion Model" expander.
    *   You'll see sliders and input boxes to configure the synthetic data:
        *   **Number of Baseline Samples**: This determines the size of your initial dataset. Leave it at the default of `1000`.
        *   **Number of Features**: How many input variables your model will use. Default `10`.
        *   **Number of Informative Features**: How many features actually influence the target. Default `3`.
        *   **Number of Redundant Features**: Features that are linear combinations of informative features. Default `0`.
        *   **Clusters per Class**: How many clusters of data points each target class has. Default `1`.
        *   **Random State**: A seed for reproducibility. Default `42`.
    *   For this exercise, you can keep all parameters at their default values to ensure a consistent starting point.

3.  **Generate Data and Train the Champion Model:**
    *   Click the **"Generate Baseline Data and Train Initial Champion Model"** button.
    *   The application will:
        *   Create a dataset based on your specified parameters.
        *   Split this data into training and testing sets.
        *   Train a Logistic Regression model (our "Champion") on the training data.
        *   Evaluate its performance on the test set.
    *   You should see success messages indicating data generation and model training.

<aside class="positive">
<b>What just happened?</b> You've established the foundation for your monitoring system. The Champion model is now your production model, and its initial performance on the baseline data serves as the benchmark against which future performance will be measured.
</aside>

4.  **Review Baseline Information:**
    *   Below the button, you'll find "Baseline Data Snapshot" showing the first few rows of your generated data, including the `target` variable.
    *   Then, you'll see a table displaying the **Baseline Metrics**:
        *   **Accuracy**: The proportion of correct predictions.
        *   **Precision**: The proportion of positive identifications that were actually correct.
        *   **Recall**: The proportion of actual positives that were identified correctly.
        *   **F1-Score**: The harmonic mean of Precision and Recall, providing a single metric that balances both.
    *   Finally, the "Baseline Target Distribution" shows the proportion of each class in your target variable, which is important for understanding the dataset balance.

## 3. Continuous Monitoring and Detecting Drift
Duration: 00:15:00

Now that our Champion model is in place, we simulate a production environment where new data arrives continuously. This is where the dashboard shines, allowing you to observe how model performance and data characteristics change over time, and to detect various forms of drift.

1.  **Configure Simulation Parameters (Introducing Drift):**
    *   Scroll down to "2. Continuous Monitoring & Alerts".
    *   Open the "Configure Simulation Parameters" expander.
    *   This section allows you to introduce different types of drift into the incoming data batches. Experiment with these sliders:
        *   **Mean Shift (Feature 0 - Data Drift)**: Shifts the average value of `feature_0`. This directly simulates a change in $P(X)$.
        *   **Std Dev Factor (Feature 0 - Data Drift)**: Changes the spread of `feature_0`. Another form of $P(X)$ change.
        *   **Concept Drift Factor**: This randomly flips a proportion of labels ($Y$) based on `feature_0`. This directly simulates a change in the relationship $P(Y|X)$, meaning the underlying "concept" the model is trying to learn has shifted.
        *   **Performance Degradation Factor**: This randomly flips a proportion of labels ($Y$) independent of features, simulating general noise or data quality issues that degrade performance.

<aside class="negative">
<b>Experimentation Tip:</b> Start by moving one slider at a time (e.g., just "Mean Shift") to observe its specific impact on the metrics before combining effects.
</aside>

2.  **Run Simulation Steps:**
    *   Below the drift parameters, you'll find "Number of steps to simulate (per click)". Set this to a reasonable number, like `10` or `20`.
    *   Click the **"Run Simulation Steps"** button.
    *   The application will generate new batches of data with the introduced drift, evaluate the Champion model on this new data, and update the monitoring plots. You'll see a success message indicating the current time step.
    *   Click this button multiple times to advance the simulation and accumulate more monitoring data.

3.  **Analyze Model Performance Over Time:**
    *   Observe the "Model Performance Over Time" plot. This shows how Accuracy, Precision, Recall, and F1-Score change as new data batches arrive.
    *   The metrics are defined as:
        *   **Accuracy:** $A = \frac{TP + TN}{TP + TN + FP + FN}$ (Total correct predictions out of all predictions)
        *   **Precision:** $P = \frac{TP}{TP + FP}$ (Correct positive predictions out of all positive predictions)
        *   **Recall:** $R = \frac{TP}{TP + FN}$ (Correct positive predictions out of all actual positives)
        *   **F1-Score:** $F1 = 2 \times \frac{P \times R}{P + R}$ (Harmonic mean of precision and recall)
    *   You'll also notice a red dashed line indicating the "Min. Accuracy Threshold". This visually helps identify when performance drops below an acceptable level.

4.  **Detecting Data Drift with K-S and JSD:**
    *   Scroll down to the "Drift Metrics Over Time (Feature 0)" plot.
    *   This plot uses statistical tests to quantify how much the distribution of `feature_0` (our simulated drifting feature) has changed from the baseline.
    *   **Kolmogorov-Smirnov (K-S) Statistic:**
        $$D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)|$$
        This measures the maximum difference between the cumulative distribution functions (CDFs) of the baseline data and the current data. A higher K-S statistic indicates greater drift.
    *   **Jensen-Shannon Divergence (JSD):**
        $$JSD(P||Q) = \frac{1}{2} D_{KL}(P||M) + \frac{1}{2} D_{KL}(Q||M)$$
        Where $M = \frac{1}{2}(P+Q)$ and $D_{KL}(P||Q) = \sum_{i} P(i) \log \left( \frac{P(i)}{Q(i)} \right)$.
        JSD measures the similarity between two probability distributions. It's bounded between 0 and 1, where 0 means the distributions are identical and 1 means they are completely different. A higher JSD indicates more drift.
    *   Orange and purple dashed lines indicate the K-S and JSD alert thresholds, respectively.

<aside class="positive">
<b>Distinguishing Drift Types:</b>
<ul>
<li>If you only introduce "Mean Shift" or "Std Dev Factor," you'll primarily see high K-S/JSD (Data Drift) and likely a drop in model performance.</li>
<li>If you only introduce "Concept Drift Factor," you might see a significant drop in model performance (P(Y|X) changed) even if K-S/JSD (P(X)) don't show much change.</li>
<li>"Performance Degradation Factor" is a general degradation; it will impact model performance metrics.</li>
</ul>
</aside>

5.  **Configure Alert Thresholds:**
    *   Open the "Configure Alert Thresholds" expander.
    *   Adjust the sliders for `Min. Accuracy Threshold`, `Max. K-S Stat Threshold`, and `Max. JSD Threshold`. These define the boundaries that trigger alerts.
    *   As you run more simulation steps, observe the "Active Alerts" section. When a metric crosses its threshold, a warning message will appear, detailing which alert was triggered and by how much. This mimics an automated alert system in a production environment.

6.  **Visualize Feature Distribution:**
    *   Below the alerts, observe the "Feature Distribution: Baseline vs. Current (Feature 0)" histogram.
    *   This visual comparison of `feature_0`'s distribution between your initial baseline and the accumulated monitoring data provides an intuitive understanding of the magnitude and direction of data drift.

## 4. Intervention Strategies: Retrain, Compare, and Promote
Duration: 00:12:00

When alerts are triggered or model health significantly deteriorates, intervention is necessary. This section demonstrates how you would address these issues by retraining, comparing models, and promoting a new Champion.

1.  **Simulate Retraining a "Challenger" Model:**
    *   Scroll to "3. Intervention Strategies".
    *   Click the **"Simulate Retraining Challenger Model"** button.
    *   The application will train a new model (our "Challenger") using all the data accumulated in the monitoring phase so far. This simulates retraining on more recent, potentially drifted data, hoping the new model learns the updated patterns.
    *   You should see a success message.

<aside class="positive">
<b>Why a Challenger?</b> In MRM, you rarely replace a production model without rigorous comparison. A "Challenger" model is trained and evaluated in parallel to ensure it truly outperforms the current "Champion" before deployment.
</aside>

2.  **Champion-Challenger Comparison:**
    *   Once a Challenger model is trained, click the **"Compare Champion vs. Challenger"** button.
    *   The dashboard will evaluate both the current Champion and the newly trained Challenger on the *most recent batch* of monitoring data.
    *   Two bar charts will appear:
        *   **Performance Metrics: Champion vs. Challenger**: Compares Accuracy, Precision, Recall, and F1-Score for both models.
        *   **Drift Metrics: Champion vs. Challenger (on Recent Data)**: Compares K-S and JSD statistics (relative to the *original* baseline) for both models. While Challenger might perform better on recent data, its drift metrics might still be high if the underlying data distribution has shifted.
    *   An info message will indicate which model (Champion or Challenger) performed better in terms of accuracy on the recent data.

3.  **Promote Challenger to Champion:**
    *   If the Challenger model demonstrates superior performance or better handles the observed drift, you might decide to promote it.
    *   Click the **"Promote Challenger to Champion"** button.
    *   The application will replace the current Champion model with the Challenger. Crucially, the monitoring history will also be reset, establishing the new Champion's performance on the *current* data as the new baseline for future monitoring.
    *   This simulates a real-world model deployment and re-initialization of monitoring.

4.  **Trigger Manual Review (Human-in-the-Loop):**
    *   Click the **"Trigger Manual Review"** button.
    *   This action simulates a critical human oversight step. Even with advanced automated monitoring, complex drift scenarios or regulatory requirements often necessitate a manual review by a risk manager or an expert.
    *   An info box will log the manual review, including details of any active alerts at that time. This emphasizes the "human-in-the-loop" aspect, a cornerstone of SR 11-7.

## 5. Conclusion: Continuous Monitoring for Trustworthy AI
Duration: 00:03:00

You've now completed a full cycle of AI model monitoring, drift detection, and intervention strategies within this simulated environment.

<aside class="positive">
<b>Key Takeaways for Risk Managers:</b>
<ul>
<li><b>Proactive Detection:</b> Continuous monitoring allows for early detection of model degradation, data drift, and concept drift before they lead to significant business impact.</li>
<li><b>Thresholds and Alerts:</b> Establishing clear thresholds and automated alerts is essential for triggering timely investigations and interventions.</li>
<li><b>Adaptive Strategies:</b> Retraining and the Champion-Challenger framework are crucial adaptive strategies to maintain model relevance and performance in dynamic environments.</li>
<li><b>Human Oversight (SR 11-7):</b> Automated tools are powerful, but human judgment, expert review, and clear governance processes are indispensable for interpreting complex situations and making informed decisions, aligning directly with regulatory expectations for trustworthy AI.</li>
</ul>
</aside>

By actively engaging with this dashboard, you've gained practical insight into the principles of AI Model Risk Management. This understanding is vital for ensuring the ongoing reliability, fairness, and transparency of AI systems, ultimately contributing to a more robust and trustworthy AI ecosystem in your organization.

Feel free to reset the page (refresh your browser) and experiment with different drift parameters and alert thresholds to observe various scenarios!
