
import streamlit as st

def on_restart_simulation_clicked():
    # Clear all session state variables to restart the application
    for key in st.session_state.keys():
        del st.session_state[key]
    st.info("Simulation has been restarted. Please navigate to the 