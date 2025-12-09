This README.md provides a comprehensive overview of the Streamlit application lab project, designed to simulate and educate on AI Model Risk Management (MRM) principles.

---

# AI Model Health Monitor: A Model Risk Management (MRM) Simulator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-link.streamlit.app/) <!-- Replace with actual deployment link if available -->

## 📊 Project Title

**AI Model Health Monitor: A Model Risk Management (MRM) Simulator for Financial Institutions**

## 📝 Project Description

This Streamlit application is an interactive lab project designed to provide a hands-on experience in Model Risk Management (MRM) for AI models, particularly relevant for professionals like **Risk Managers** in financial institutions. It simulates the lifecycle of an AI model in a production environment, focusing on continuous monitoring, detection of performance degradation and various types of data/concept drift, and strategic interventions like model retraining and promotion.

The simulator guides users through a structured workflow, from establishing a model's baseline to managing its operational health over time, reflecting principles adapted from regulatory frameworks such as **SR 11-7**. Users will learn to identify key indicators of model instability, configure alert thresholds, evaluate challenger models, and exercise human-in-the-loop governance for critical decisions.

This project aims to bridge the gap between theoretical MRM concepts and practical application, allowing users to visualize model behavior under various realistic scenarios and reinforce best practices for trustworthy AI deployment.

## ✨ Features

The application offers a rich set of features to simulate a comprehensive MRM workflow:

*   **Guided Workflow**: A multi-page Streamlit interface that walks the user through the entire MRM lifecycle, from baseline establishment to conclusion.
*   **Synthetic Data Generation**: Generates synthetic classification datasets with configurable parameters to simulate various real-world scenarios, including:
    *   **Performance Degradation**: Direct degradation of model accuracy by flipping labels.
    *   **Data Drift (Covariate Shift)**: Shifts in input feature distributions (e.g., mean, standard deviation).
    *   **Concept Drift**: Changes in the underlying relationship between features and the target variable.
*   **Champion Model Training & Deployment**: Simulates the initial training and deployment of a "Champion" Logistic Regression model on baseline data.
*   **Continuous Monitoring Dashboard**:
    *   Tracks key **Performance Metrics** over simulated time steps:
        *   **Accuracy**: $$A = \frac{TP + TN}{TP + TN + FP + FN}$$
        *   **Precision**: $$P = \frac{TP}{TP + FP}$$
        *   **Recall (Sensitivity)**: $$R = \frac{TP}{TP + FN}$$
        *   **F1-Score**: $$F1 = 2 \times \frac{P \times R}{P + R}$$
    *   Monitors **Data Drift Metrics** for selected features (e.g., `feature_0`):
        *   **Kolmogorov-Smirnov (K-S) Statistic**: Measures the maximum difference between cumulative distribution functions. $$D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)|$$
        *   **Jensen-Shannon Divergence (JSD)**: Measures the similarity between two probability distributions. $$\text{JSD}(P||Q) = \sqrt{\frac{1}{2} D_{\text{KL}}(P||M) + \frac{1}{2} D_{\text{KL}}(Q||M)}$$ where $M = \frac{1}{2}(P+Q)$.
*   **Dynamic Alert Configuration**: Allows users to interactively set and adjust alert thresholds for accuracy, K-S statistic, and JSD, observing their immediate impact on triggered alerts.
*   **Challenger Model Retraining**: Simulates retraining a new "Challenger" model on updated, potentially drifted production data.
*   **Champion-Challenger Comparison**: Provides interactive bar charts to compare the performance and drift metrics of the current Champion model against the newly trained Challenger model.
*   **Model Promotion**: Simulates the decision-making process to promote a superior Challenger model to become the new Champion.
*   **Human-in-the-Loop Governance**: Features a manual review logging system to emphasize the importance of human oversight and auditability in critical MRM decisions.
*   **Interactive Visualizations**: Utilizes Plotly for dynamic and informative graphs, including time-series plots of metrics and overlaid histograms for data distribution shifts.

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Ensure you have Python 3.8 or higher installed.

*   [Python](https://www.python.org/downloads/) (3.8+)
*   [pip](https://pip.pypa.io/en/stable/installation/) (usually comes with Python)
*   [Git](https://git-scm.com/downloads) (for cloning the repository)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/ai-model-health-monitor.git
    cd ai-model-health-monitor
    ```
    (Replace `your-username/ai-model-health-monitor` with the actual repository URL)

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required packages**:
    Create a `requirements.txt` file in the root directory with the following content:
    ```
    streamlit
    pandas
    numpy
    scikit-learn
    scipy
    plotly
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♀️ Usage

To run the Streamlit application:

1.  **Activate your virtual environment** (if not already active):
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

2.  **Navigate to the project root directory** (where `app.py` is located):
    ```bash
    cd /path/to/ai-model-health-monitor
    ```

3.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

    Your default web browser should automatically open a new tab with the application running (usually at `http://localhost:8501`). If not, copy and paste the provided URL from your terminal into your browser.

## 📁 Project Structure

The project is organized into modular components for clarity and maintainability:

```
ai-model-health-monitor/
├── app.py                            # Main Streamlit application entry point and navigation logic
├── utils.py                          # Utility functions for data generation, model training, metrics, and monitoring simulation
├── application_pages/                # Directory containing individual Streamlit pages
│   ├── page_01_welcome.py            # Welcome screen and project introduction
│   ├── page_02_establishing_the_baseline.py # Data baseline generation and review
│   ├── page_03_training_the_champion.py     # Champion model training simulation
│   ├── page_04_reviewing_baseline_performance.py # Champion model baseline performance review
│   ├── page_05_initial_stable_monitoring.py  # Initial stable monitoring phase
│   ├── page_06_detecting_performance_degradation.py # Simulates and detects performance degradation
│   ├── page_07_detecting_data_drift.py        # Explores and detects data drift (covariate shift)
│   ├── page_08_detecting_concept_drift.py     # Explores and detects concept drift
│   ├── page_09_configuring_alert_thresholds.py # Interactive alert threshold configuration
│   ├── page_10_retraining_a_challenger.py     # Challenger model retraining simulation
│   ├── page_11_champion_challenger_comparison.py # Comparison and promotion of models
│   ├── page_12_human_in_the_loop_governance.py # Manual review and governance simulation
│   └── page_13_conclusion.py         # Project conclusion and restart option
└── requirements.txt                  # List of Python dependencies
```

## 🛠️ Technology Stack

*   **Python**: The core programming language (version 3.8+).
*   **Streamlit**: For building the interactive web application and user interface.
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical operations, especially with arrays.
*   **Scikit-learn**: For machine learning functionalities (data generation, model training, metrics).
*   **SciPy**: For scientific computing, particularly statistical tests (K-S statistic) and mathematical functions (KL divergence for JSD).
*   **Plotly**: For creating interactive and publication-quality visualizations.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 📧 Contact

For any questions or inquiries, please reach out to:

*   **Project Maintainer**: [Your Name/Organization Name]
*   **Email**: [your.email@example.com]
*   **GitHub**: [Your GitHub Profile/Organization Profile]

---