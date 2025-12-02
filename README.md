# AI Model Health Dashboard Simulator

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

**AI Model Health Dashboard Simulator: An Interactive Lab for Model Risk Management (MRM)**

This Streamlit application provides an interactive, simulated environment for exploring and understanding the critical aspects of AI Model Health Monitoring. Designed primarily for **Risk Managers** and AI/ML practitioners, it offers a hands-on experience in implementing key Model Risk Management (MRM) principles, especially those aligning with regulatory guidance like **SR 11-7** for adaptive AI systems.

The simulator allows users to:
*   **Generate synthetic data** and train an initial "Champion" model.
*   **Simulate continuous monitoring** of model performance and data characteristics over time.
*   **Introduce various types of drift** (data drift, concept drift) and performance degradation.
*   **Configure and observe threshold-based alerts** for deviations from acceptable model health.
*   **Practice intervention strategies** such as retraining "Challenger" models, comparing them against the Champion, and promoting superior models.
*   **Understand the necessity of human-in-the-loop** processes for qualitative assessment and governance.

By interactively manipulating parameters and observing their impact, users can gain a practical understanding of how to proactively manage risks associated with AI models in dynamic real-world scenarios.

## Features

The dashboard offers the following key functionalities:

*   **Synthetic Data Generation**:
    *   Generate baseline classification datasets with configurable parameters (number of samples, features, informative features, clusters per class).
    *   Introduce controlled **data drift** (mean shift, standard deviation factor for features).
    *   Simulate **concept drift** (flipping labels based on feature values).
    *   Introduce **performance degradation** (random label flipping).
*   **Champion Model Training**:
    *   Train an initial Logistic Regression "Champion" model on the baseline data.
    *   Evaluate and display baseline performance metrics (Accuracy, Precision, Recall, F1-Score).
*   **Continuous Monitoring**:
    *   Simulate new data batches arriving over time, applying configured drift/degradation.
    *   Track and visualize historical trends for performance metrics (Accuracy, Precision, Recall, F1-Score).
    *   Track and visualize historical trends for data drift metrics (Kolmogorov-Smirnov Statistic, Jensen-Shannon Divergence for Feature 0).
*   **Alerting System**:
    *   Configurable thresholds for minimum accuracy, maximum K-S statistic, and maximum JSD.
    *   Trigger and display alerts when monitored metrics cross predefined thresholds.
*   **Intervention Strategies**:
    *   **Challenger Model Retraining**: Train a new "Challenger" model on accumulated recent monitoring data.
    *   **Champion-Challenger Comparison**: Visually compare the Champion and Challenger models' performance and drift metrics on recent data.
    *   **Model Promotion**: Promote the Challenger model to Champion if its performance is superior, effectively simulating model redeployment.
    *   **Manual Review Trigger**: Simulate initiating a human-in-the-loop review process, emphasizing human oversight as per SR 11-7.
*   **Interactive Visualizations**:
    *   Line plots showing model performance and drift metrics over time.
    *   Bar charts for Champion vs. Challenger comparisons.
    *   Histograms to compare baseline and current feature distributions, aiding in data drift analysis.
*   **Educational Content**: Inline explanations of key concepts like data drift, concept drift, and MRM principles.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository** (if hosted on GitHub, otherwise download the source code):
    ```bash
    git clone https://github.com/your-username/ai-model-health-dashboard.git
    cd ai-model-health-dashboard
    ```
    *(Replace `https://github.com/your-username/ai-model-health-dashboard.git` with the actual repository URL)*

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
    Create a `requirements.txt` file in the root directory of your project with the following content:

    ```
    streamlit==1.30.0 # Or compatible version
    pandas
    numpy
    scikit-learn
    scipy
    plotly
    ```

    Then install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application**:
    Ensure your virtual environment is activated and you are in the project's root directory.
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your default web browser.

2.  **Navigate the Dashboard**:
    *   The application starts on an introductory page. Use the sidebar to navigate to the "Dashboard" page.
    *   **Section 1: Baseline Setup**:
        *   Configure parameters for synthetic data generation.
        *   Click "Generate Baseline Data and Train Initial Champion Model" to set up your initial environment. This trains the first model and displays its baseline performance.
    *   **Section 2: Continuous Monitoring & Alerts**:
        *   Adjust the sliders under "Configure Simulation Parameters" to introduce mean shift, standard deviation changes (data drift), concept drift, or performance degradation.
        *   Set the number of steps to simulate and click "Run Simulation Steps" to advance the simulation.
        *   Observe how model performance and drift metrics change over time in the plots.
        *   Adjust "Alert Thresholds" to see how they trigger warnings.
    *   **Section 3: Intervention Strategies**:
        *   If performance degrades or alerts are triggered, click "Simulate Retraining Challenger Model".
        *   Use "Compare Champion vs. Challenger" to evaluate the new model.
        *   If the Challenger is superior, click "Promote Challenger to Champion" to update the active model and reset monitoring.
        *   Click "Trigger Manual Review" to simulate a human intervention.

## Project Structure

```
.
├── app.py                      # Main Streamlit application entry point
├── application_pages/          # Directory for individual Streamlit pages/modules
│   └── dashboard.py            # Core logic and UI for the AI Model Health Dashboard
├── requirements.txt            # List of Python dependencies
└── README.md                   # Project documentation (this file)
```

*   `app.py`: This file acts as the primary launcher for the Streamlit application. It sets up the page configuration, displays initial introductory text, and handles navigation to different application pages (in this case, `dashboard.py`). It also contains all utility functions for global availability, fulfilling a common Streamlit app structure requirement.
*   `application_pages/dashboard.py`: This module contains the main logic, UI components, and interactivity for the AI Model Health Dashboard itself. It defines how the baseline is set, monitoring is performed, and interventions are handled.

## Technology Stack

*   **Python 3.8+**: The core programming language.
*   **Streamlit**: For creating interactive web applications with pure Python.
*   **Pandas**: For data manipulation and analysis, especially with dataframes.
*   **NumPy**: For numerical operations, particularly array manipulation and mathematical functions.
*   **Scikit-learn**: For machine learning tasks, including synthetic data generation (`make_classification`), model training (`LogisticRegression`), and performance metric calculation (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`).
*   **SciPy**: For scientific computing, specifically statistical tests (`ks_2samp`) and distance metrics (`jensenshannon`).
*   **Plotly**: For creating interactive and publication-quality statistical graphics.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes and commit them (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

Please ensure your code adheres to good practices and includes appropriate documentation and tests where applicable.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if applicable, otherwise state "No specific license defined.").

## Contact

For any questions, feedback, or further information, please reach out to:

*   **QuantUniversity (QuLab)**: [https://www.quantuniversity.com/](https://www.quantuniversity.com/)
*   You can open an issue on the GitHub repository for any specific project-related queries.
