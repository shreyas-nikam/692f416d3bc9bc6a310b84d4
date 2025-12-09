
import streamlit as st

st.set_page_config(page_title="QuLab - AI Model Health Monitor", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("AI Model Health Monitor")

# Initialize session state for navigation and model status if not already set
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Welcome & Your Role"
if 'champion_model_name' not in st.session_state:
    st.session_state.champion_model_name = "Logistic Regression"
if 'model_health_status' not in st.session_state:
    st.session_state.model_health_status = "Stable"

# Update model health status dynamically based on current alerts
if st.session_state.get('historical_metrics') and st.session_state.historical_metrics:
    latest_step = st.session_state.historical_metrics[-1]
    if latest_step['alerts']:
        if 'accuracy' in latest_step['alerts'] and ('ks_feature_0' in latest_step['alerts'] or 'jsd_feature_0' in latest_step['alerts']):
            st.session_state.model_health_status = "Degradation & Drift Detected"
        elif 'accuracy' in latest_step['alerts']:
            st.session_state.model_health_status = "Degradation Detected"
        elif 'ks_feature_0' in latest_step['alerts'] or 'jsd_feature_0' in latest_step['alerts']:
            st.session_state.model_health_status = "Drift Warning"
    else:
        st.session_state.model_health_status = "Stable"


st.sidebar.markdown(f"**Role:** Risk Manager")
st.sidebar.markdown(f"**Current Champion Model:** {st.session_state.champion_model_name}")
st.sidebar.markdown(f"**Model Health Status:** {st.session_state.model_health_status}")
st.sidebar.divider()

# Define pages for navigation
page_options = [
    "Welcome & Your Role",
    "Establishing the Baseline",
    "Training the Champion",
    "Reviewing Baseline Performance",
    "Initial Stable Monitoring",
    "Detecting Performance Degradation",
    "Detecting Data Drift",
    "Detecting Concept Drift",
    "Configuring Alert Thresholds",
    "Retraining a Challenger",
    "Champion-Challenger Comparison",
    "Human-in-the-Loop & Governance",
    "Conclusion"
]

# Dynamic navigation based on current_page in session state
# The selectbox will always reflect the current page from session_state
selected_page_index = page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0

page = st.sidebar.selectbox(
    label="Navigation",
    options=page_options,
    index=selected_page_index,
    key="sidebar_navigation"
)

# Update st.session_state.current_page if the user manually changes the selectbox
if page != st.session_state.current_page:
    st.session_state.current_page = page
    st.rerun()

st.divider()

# Route to the selected page
if st.session_state.current_page == "Welcome & Your Role":
    from application_pages.page_01_welcome import main
    main()
elif st.session_state.current_page == "Establishing the Baseline":
    from application_pages.page_02_establishing_the_baseline import main
    main()
elif st.session_state.current_page == "Training the Champion":
    from application_pages.page_03_training_the_champion import main
    main()
elif st.session_state.current_page == "Reviewing Baseline Performance":
    from application_pages.page_04_reviewing_baseline_performance import main
    main()
elif st.session_state.current_page == "Initial Stable Monitoring":
    from application_pages.page_05_initial_stable_monitoring import main
    main()
elif st.session_state.current_page == "Detecting Performance Degradation":
    from application_pages.page_06_detecting_performance_degradation import main
    main()
elif st.session_state.current_page == "Detecting Data Drift":
    from application_pages.page_07_detecting_data_drift import main
    main()
elif st.session_state.current_page == "Detecting Concept Drift":
    from application_pages.page_08_detecting_concept_drift import main
    main()
elif st.session_state.current_page == "Configuring Alert Thresholds":
    from application_pages.page_09_configuring_alert_thresholds import main
    main()
elif st.session_state.current_page == "Retraining a Challenger":
    from application_pages.page_10_retraining_a_challenger import main
    main()
elif st.session_state.current_page == "Champion-Challenger Comparison":
    from application_pages.page_11_champion_challenger_comparison import main
    main()
elif st.session_state.current_page == "Human-in-the-Loop & Governance":
    from application_pages.page_12_human_in_the_loop_governance import main
    main()
elif st.session_state.current_page == "Conclusion":
    from application_pages.page_13_conclusion import main
    main()
