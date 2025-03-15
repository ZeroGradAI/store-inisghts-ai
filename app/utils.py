import streamlit as st
import time

def update_analysis_status(in_progress=False):
    """Update the analysis status in session state."""
    if 'analysis_in_progress' not in st.session_state:
        st.session_state.analysis_in_progress = False
    if 'last_analysis_time' not in st.session_state:
        st.session_state.last_analysis_time = None
        
    st.session_state.analysis_in_progress = in_progress
    if not in_progress:
        st.session_state.last_analysis_time = time.strftime("%I:%M:%S %p") 