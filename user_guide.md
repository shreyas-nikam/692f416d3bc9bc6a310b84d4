id: 692f416d3bc9bc6a310b84d4_user_guide
summary: AI Design and Deployment Lab 4 User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# AI Model Health Monitor: A Codelab for Risk Managers

## Welcome & Your Role
Duration: 0:02:00

<aside class="positive">
This codelab is designed for **Risk Managers** and anyone interested in the practical aspects of AI Model Risk Management (MRM). It focuses on understanding and responding to critical model health issues in a simulated production environment.
</aside>

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

Ready to ensure the trustworthiness of our AI models? Click the "Start My Monitoring Journey" button in the application to begin.

## Establishing the Baseline
Duration: 0:01:30

Before deploying any AI model into production, it is crucial for a Risk Manager to thoroughly understand its foundational characteristics. This includes defining the initial training data, its distribution, and the expected performance benchmarks. This "baseline" serves as the standard against which all future model performance and data shifts will be measured.

In this step, we will simulate the generation of a synthetic dataset that represents the kind of data our AI model (a binary classifier in this case) would be trained on. This dataset will inform our understanding of the model's initial learning environment.

### Persona's Action: Reviewing Initial Data Integrity
As a Risk Manager, your first task is to ensure the integrity and representativeness of the data used for model training. Understanding the features and target distribution helps you anticipate potential biases or imbalances, which are critical inputs for your initial risk assessment.

In the application, you will see a "Glimpse of Baseline Data", showing the first few rows of the synthetic dataset. Below that, the data shape and the distribution of the target variable are displayed. This initial overview confirms that the data is structured as expected and provides the necessary foundation for training our Champion model.

<aside class="positive">
<b>Baseline data is your golden standard.</b> Always keep a clear record of the initial training data's characteristics to compare against future production data.
</aside>

Having verified the baseline data, the next step in your workflow is to oversee the training of the initial model that will be deployed. Click the "Proceed to Train the Champion Model" button in the application.

## Training the Champion
Duration: 0:01:00

With the baseline data established and reviewed, the next logical step for a Risk Manager is to oversee the training of the initial model, which we will designate as our "Champion" model. This model represents the version currently deemed fit for production, and its performance will be continuously monitored.

### Persona's Action: Initiating Model Deployment Preparation
Your role here is to ensure that the model is trained on the approved baseline data, setting the stage for its deployment. This is a critical step in the AI model lifecycle, as the Champion model will serve as the benchmark for all subsequent evaluations and challenger comparisons.

The application will automatically train a Logistic Regression model using the baseline data. A success message "Champion Model trained successfully" will confirm this. This model will be the one whose health you will track throughout the simulation.

The Champion model has been successfully trained on our baseline dataset. Now, it's imperative to review its initial performance to set realistic expectations for its behavior in production. This involves calculating key classification metrics that will guide our monitoring efforts. Click the "Review Champion Model's Baseline Performance" button in the application.

## Reviewing Baseline Performance
Duration: 0:03:00

As a Risk Manager, a thorough understanding of the Champion model's initial performance is non-negotiable. These baseline metrics serve as the crucial benchmark against which all future model behavior in production will be evaluated. Any significant deviation from these established levels will trigger alerts and require investigation.

### Persona's Action: Establishing Performance Benchmarks
Your task here is to meticulously review the calculated performance metrics, internalizing what constitutes "normal" and "acceptable" model behavior. This forms the foundation for setting robust monitoring thresholds and risk tolerance levels.

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

In the application, you will see these four metrics displayed as `st.metric` widgets, showing the Champion model's baseline performance. These values are your reference points for "healthy" model operation.

<aside class="negative">
Remember: A single metric rarely tells the whole story. A holistic view using multiple metrics is essential for a complete model risk assessment.
</aside>

These baseline metrics represent the Champion model's expected performance under ideal, untainted conditions. They provide a critical reference point for continuous monitoring. Any significant drift from these values during production could signal model degradation or data integrity issues, necessitating your immediate attention.

Now that we have established the performance baseline, it's time to simulate the model's deployment and initiate continuous monitoring to ensure its ongoing health and reliability. Click the "Start Initial Stable Monitoring" button in the application.

## Initial Stable Monitoring
Duration: 0:02:30

As a Risk Manager, your commitment to continuous validation begins now. The Champion model, having been trained and its baseline performance documented, is now in production. Your immediate task is to observe its behavior under normal operating conditions to confirm its stability before any significant external factors come into play.

### Persona's Action: Verifying Operational Stability
In this phase, you are looking for consistency. The model should maintain its performance, and the incoming data should closely resemble the baseline. This initial period of stable monitoring builds confidence in the model's robustness and helps validate your baseline assumptions. It's about establishing a "normal" operational rhythm for your monitoring dashboard.

The application will simulate 5 initial stable monitoring steps. During these steps, new data batches are generated, and the Champion model's performance and data drift metrics are calculated against the baseline. You will see two interactive plots:

1.  **Model Performance Metrics Over Time:** This plot tracks Accuracy, Precision, Recall, and F1-Score. A red dashed line indicates a predefined minimum acceptable accuracy threshold.
2.  **Data Drift Metrics Over Time (Feature 0):** This plot tracks the Kolmogorov-Smirnov (K-S) Statistic and Jensen-Shannon Divergence (JSD) for a specific feature (Feature 0). Orange and purple dashed lines indicate maximum tolerable thresholds for these drift metrics.

During these initial steps, you should observe that both performance and drift metrics remain within acceptable bounds, reaffirming the model's health post-deployment. This provides a clean slate before potential issues arise.

<aside class="positive">
A stable monitoring period is crucial to confirm your deployment was successful and to validate your initial monitoring setup before real-world challenges appear.
</aside>

These initial monitoring steps demonstrate the model operating under stable conditions. Both performance and data drift metrics remain within acceptable bounds, reaffirming the model's health post-deployment. However, the real world is dynamic. Your vigilance as a Risk Manager is paramount as conditions change. Click the "Simulate Performance Degradation" button in the application.

## Detecting Performance Degradation
Duration: 0:02:00

As a Risk Manager, you know that models, like any other asset, can degrade over time. This page simulates a scenario where our Champion model's performance begins to decline in production. This degradation can be due to various reasons, such as shifts in customer behavior, changes in external economic factors, or even subtle issues within data pipelines.

### Persona's Action: Identifying a Performance Issue
Your objective is to quickly identify when the model's performance falls below acceptable thresholds. This early detection allows for timely intervention, preventing significant financial or operational impact. You will observe how the performance metrics, particularly accuracy, start to trend downwards, signaling a critical change in the model's effectiveness.

The application will simulate additional monitoring steps (up to Time Step 15), gradually introducing a "performance degradation factor." This directly impacts the model's ability to make correct predictions, causing its performance metrics to decline.

Observe the **Model Performance Metrics Over Time** plot. You should now see the `Accuracy` line (blue) trending downwards and potentially crossing the `Min. Accuracy Threshold` (red dashed line). The `Data Drift Metrics Over Time` plot for Feature 0, however, should remain relatively stable, indicating that the degradation is due to factors other than direct input data shifts in this specific scenario.

<aside class="negative">
Performance degradation can be subtle at first. Early detection relies on robust monitoring and appropriately set alert thresholds.
</aside>

Performance degradation is a clear signal that something is amiss. As a Risk Manager, your next step is to understand if this degradation is accompanied by shifts in the input data itself, which is often a root cause. Click the "Investigate Data Drift" button in the application.

## Understanding and Detecting Data Drift (Covariate Shift)
Duration: 0:04:00

As a Risk Manager, recognizing data drift is paramount. Data drift, or covariate shift, occurs when the statistical properties of the input features change over time. This means the data that your model is seeing in production is different from the data it was trained on. This can significantly degrade model performance, even if the underlying relationship between features and the target remains the same.

### Persona's Action: Diagnosing Data Integrity Issues
Your role is to identify these shifts in input data distributions. Understanding *how* the data has changed provides crucial context for observed performance degradation and informs decisions about model retraining or data pipeline adjustments. You will use statistical measures and visual comparisons to confirm the presence and magnitude of data drift.

### Defining and Calculating Data Drift Metrics
We will use two key metrics to quantify data drift:

*   **Kolmogorov-Smirnov (K-S) Statistic:** Measures the maximum difference between the cumulative distribution functions of two samples. A higher K-S statistic indicates a greater difference between the baseline and current data distributions.
    $$D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)|$$
    Where $F_{1,n}(x)$ and $F_{2,m}(x)$ are the empirical distribution functions for the two samples.

*   **Jensen-Shannon Divergence (JSD):** A symmetric and smoothed version of Kullback-Leibler (KL) Divergence. It measures the similarity between two probability distributions. A higher JSD value indicates greater dissimilarity or divergence between the baseline and current data distributions.
    $$\text{JSD}(P||Q) = \frac{1}{2} D_{\text{KL}}(P||M) + \frac{1}{2} D_{\text{KL}}(Q||M)$$
    Where $M = \frac{1}{2}(P+Q)$, and $D_{\text{KL}}(P||Q) = \sum_i P(i) \log\left(\frac{P(i)}{Q(i)}\right)$ is the Kullback-Leibler Divergence.

Higher values for both K-S and JSD indicate that the current data's distribution has diverged significantly from the baseline, signaling data drift.

### Example: Demonstrating K-S and JSD
The application provides a quick example of K-S and JSD calculations for simple synthetic data:
-   `K-S (no drift): 0.0900, JSD (no drift): 0.0001`
-   `K-S (with drift): 0.5000, JSD (with drift): 0.0573`

The example demonstrates that even subtle shifts in data distributions lead to higher K-S and JSD statistics, indicating potential data drift. This aligns with your goal of identifying these changes as a Risk Manager.

The application will now simulate more monitoring steps (up to Time Step 30), specifically introducing a `mean_shift` to `feature_0`, causing data drift. The direct performance degradation from the previous step is reset to isolate the effect of data drift.

Observe the **Data Drift Metrics Over Time (Feature 0)** plot. You should see both the `K-S Statistic` and `JSD` lines increasing significantly and likely crossing their respective alert thresholds. The `Model Performance Metrics` may also show a further decline, as data drift often leads to degraded performance.

Below the plots, a **Visualizing Feature 0 Distribution Shift** section shows overlaid histograms of the baseline Feature 0 data and the most recent batch of monitoring data. This visual comparison provides compelling evidence of the data drift.

<aside class="positive">
Visualizing feature distributions alongside statistical metrics provides a powerful, intuitive way to understand data drift.
</aside>

Detecting data drift is a critical skill for a Risk Manager. It indicates that the model is now operating on data fundamentally different from its training environment, often leading to reduced trustworthiness and accuracy. The next challenge is to understand if the *relationship* between features and the target is changing, which is known as concept drift. Click the "Explore Concept Drift" button in the application.

## Understanding and Detecting Concept Drift
Duration: 0:02:30

As a Risk Manager, you understand that data drift is one challenge, but a more insidious one is **concept drift**. This occurs when the underlying relationship between the input features and the target variable changes over time. Unlike data drift, where the input distribution changes but the output relationship remains stable, concept drift means the very "concept" the model learned is no longer valid.

### Persona's Action: Uncovering Model Irrelevance
Your objective here is to identify if the model's fundamental understanding of the world has become outdated. This often manifests as a decline in model performance even when input data distributions appear stable or when the degradation is more severe than data drift alone would explain. Detecting concept drift requires vigilance and careful analysis of performance trends.

The application will now simulate additional monitoring steps (up to Time Step 45), introducing a `concept_drift_factor`. This means the way `feature_0` influences the `target` variable changes. Both direct performance degradation and data drift from previous steps are reset to focus on concept drift.

Observe the **Model Performance Metrics Over Time** plot. You will likely see a significant decline in `Accuracy` and other performance metrics, indicating the model's learned concept is failing. Crucially, now observe the **Data Drift Metrics Over Time (Feature 0)** plot. In this focused scenario, the K-S and JSD statistics for Feature 0 should remain relatively stable and below their thresholds, or at least not show the same drastic increase as during data drift. This distinction is vital for a Risk Manager: performance degradation with stable input data distributions strongly suggests concept drift.

<aside class="negative">
Concept drift is harder to diagnose than data drift, as the input data itself may appear normal. Performance metrics are key indicators.
</aside>

Having observed various forms of model degradation and drift, it's clear that continuous monitoring requires pre-defined alert thresholds. Your ability to configure these thresholds appropriately is key to proactive risk management. Click the "Configure Alert Thresholds" button in the application.

## Configuring Alert Thresholds
Duration: 0:03:00

As a Risk Manager, your ability to define and adjust alert thresholds is central to proactive AI model risk management. These thresholds are not arbitrary; they reflect the organization's risk appetite and the criticality of the model's function. Setting them appropriately ensures that genuine issues are flagged without creating excessive noise from false positives.

### Persona's Action: Defining Risk Boundaries
In this step, you will actively configure the boundaries for acceptable model performance and data stability. Adjusting these sliders allows you to immediately see how different risk tolerances would have impacted the detection of issues during the simulated monitoring period. This hands-on experience reinforces the practical implications of your risk management decisions.

In the application, you will find three sliders:
-   **Min. Accuracy Threshold $\Delta A_{min}$**: Set the minimum acceptable accuracy.
-   **Max. K-S Statistic Threshold (Feature 0) $\Delta KS_{max}$**: Set the maximum tolerable data drift by K-S.
-   **Max. JSD Threshold (Feature 0) $\Delta JSD_{max}$**: Set the maximum tolerable data drift by JSD.

Adjust these sliders. As you do, observe how the horizontal dashed lines on both the performance and drift plots shift. Additionally, 'X' markers will appear on the plots at time steps where the metrics cross your *newly set* thresholds, indicating triggered alerts. The **Alert Log** text area below the plots will also dynamically update to reflect all alerts detected based on your current threshold settings.

<aside class="positive">
Fine-tuning alert thresholds is an iterative process. It's about balancing sensitivity to real issues with avoiding alert fatigue.
</aside>

By interactively setting these thresholds, you gain a deeper appreciation for the balance between sensitivity to change and avoiding alert fatigue. Now that you have configured your monitoring, it's time to consider intervention strategies when alerts are triggered. Click the "Proceed to Retrain a Challenger Model" button in the application.

## Simulating Interventions: Retraining a "Challenger" Model
Duration: 0:02:00

As a Risk Manager, detecting performance degradation or data drift is only the first step. The next critical action is intervention. When monitoring alerts signal that a Champion model is no longer fit for purpose, the standard protocol is to retrain a new model, often referred to as a "Challenger." This new model is trained on up-to-date data, including the recent production data that caused the Champion's issues.

### Persona's Action: Initiating a Corrective Action
Your role here is to trigger the retraining process. This decision is informed by the insights gathered from continuous monitoring and alert configurations. The Challenger model represents a potential solution to mitigate the risks associated with the Champion's declining health. Once trained, it will be rigorously compared against the current Champion to determine its suitability for deployment.

In the application, click the "Simulate Retraining Challenger Model" button. The simulator will train a new Challenger Logistic Regression model. This Challenger model is trained on a combined dataset of the original baseline data and *all* the accumulated monitoring data up to the current time step. This is a common strategy to ensure the new model learns from the most recent data distributions and concepts.

After retraining, the Challenger model's performance on the *most recent batch* of monitoring data will be displayed using the same classification metrics (Accuracy, Precision, Recall, F1-Score). This gives you an initial indication of its potential effectiveness.

<aside class="positive">
Retraining on recent data is a primary strategy to combat model degradation and drift, ensuring the model remains relevant to current conditions.
</aside>

The Challenger Model has been trained. You can now compare it with the Champion. Click the "Proceed to Champion-Challenger Comparison" button in the application.

## Champion-Challenger Comparison and Model Promotion
Duration: 0:03:00

As a Risk Manager, after a Challenger model has been retrained, your crucial next step is to rigorously compare its performance against the current Champion model. This comparison must be data-driven, using the most recent and relevant production data to assess which model is truly superior. The goal is to ensure that any model promoted to Champion status genuinely mitigates the previously identified risks.

### Persona's Action: Making a Data-Driven Deployment Decision
You are now in the decision-making phase. Based on the comparative analysis of performance and drift metrics, you must decide whether the Challenger model offers sufficient improvement to warrant replacing the current Champion. This is a high-stakes decision that directly impacts the ongoing reliability and trustworthiness of our AI system.

In the application, first, ensure the Challenger model has been trained (from the previous step). Then, click the "Compare Champion vs. Challenger" button. The application will evaluate both the current Champion and the newly trained Challenger models on the *entire accumulated monitoring data*. This provides a comprehensive view of how each model performs under the conditions that caused the Champion's degradation.

You will see two bar charts:
-   **Champion vs. Challenger Performance Comparison:** This chart compares Accuracy, Precision, Recall, and F1-Score for both models. Ideally, the Challenger should show significantly higher scores on the recent, challenging data.
-   **Champion vs. Challenger Data Drift Metrics Comparison (Feature 0):** This chart will show the K-S and JSD statistics. While these metrics primarily reflect the *data itself* rather than model performance, they provide context. If data drift was a primary cause of degradation, the Challenger model, being trained on this new data, should theoretically perform better in its presence.

<aside class="positive">
Always compare models on the *same, recent, challenging data* to make an informed decision about promotion. This reveals which model is better adapted to current realities.
</aside>

The comparison plots provide a clear overview. If the Challenger demonstrates superior performance, especially on the data where the Champion struggled, it indicates that retraining has been effective. Your final decision as a Risk Manager is to authorize its promotion to production. If the Challenger performs better, the "Promote Challenger to Champion" button will become enabled. Clicking it will replace the old Champion with the new Challenger, and the monitoring simulation will reset, restarting with the new Champion.

After reviewing the comparison and making your decision (or if you choose not to promote), click the "Proceed to Human-in-the-Loop & Governance" button in the application.

## Human-in-the-Loop and Governance in AI Monitoring (SR 11-7)
Duration: 0:02:00

As a Risk Manager, while automated monitoring and model retraining are powerful tools, they are not a substitute for human oversight. Especially for critical AI models in regulated environments, a "human-in-the-loop" (HITL) approach is essential. This ensures that complex alerts, ambiguous situations, or high-impact decisions are subjected to qualitative assessment, expert judgment, and adherence to governance protocols, as emphasized by regulatory guidance like SR 11-7.

### Persona's Action: Exercising Prudent Oversight
Your role is to activate manual review processes when automated systems signal issues that require deeper investigation or strategic decisions. This demonstrates the critical balance between automation and human intelligence in maintaining trustworthy AI. It's about ensuring that accountability remains firmly with human experts, especially when model decisions have significant consequences.

In the application, click the "Trigger Manual Review" button. This simulates a risk manager initiating a manual investigation. An informational message will appear, and an entry will be added to the "Manual Review Log" text area, recording the time step and any active alerts.

<aside class="negative">
Even the best automated systems can miss nuanced issues. Human judgment, domain expertise, and a clear governance framework are indispensable for comprehensive AI risk management.
</aside>

This page underscores that even with the most sophisticated monitoring and automated interventions, human judgment and governance remain indispensable in the AI model lifecycle. Your final responsibility is to ensure that these processes are robust and auditable. Click the "Proceed to Conclusion" button in the application.

## Conclusion: Continuous Monitoring for Trustworthy AI
Duration: 0:01:30

Congratulations, Risk Manager! You have successfully navigated the dynamic landscape of AI model health, from establishing baselines and detecting various forms of drift and degradation to configuring alerts and orchestrating interventions. This simulated journey has reinforced the critical importance of a robust AI Model Risk Management (MRM) framework, particularly one adapted from principles like SR 11-7.

### Key Takeaways for Your Role:
-   **Proactive Detection:** Continuous monitoring is not merely reactive; it's about proactively identifying subtle shifts before they escalate into significant risks.
-   **Adaptive Controls:** Alert thresholds are dynamic and should reflect the evolving risk appetite and model criticality. Your ability to adjust them is a key control mechanism.
-   **Champion-Challenger Framework:** The iterative process of retraining and comparing models ensures that your organization can adapt to new data environments and maintain optimal performance.
-   **Human-in-the-Loop:** Even with advanced automation, human expertise and judgment are irreplaceable for critical decisions, policy enforcement, and addressing nuanced risks.
-   **Governance and Accountability:** Every step, from data validation to model promotion, requires clear governance protocols and auditable records to ensure compliance and build trust.

By applying these principles, you contribute directly to building and maintaining trustworthy and reliable AI systems within our financial institution. Your vigilance is a cornerstone of responsible AI deployment.

Thank you for your dedication to sound AI Model Risk Management!

If you wish to explore the simulation again with different parameters or simply restart your journey, click the "Restart Simulation" button in the application.
