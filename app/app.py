import streamlit as st
import os
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import torch
import sys

# Add the model directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="Store Insights AI",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'gender_analysis_done' not in st.session_state:
    st.session_state.gender_analysis_done = False
    
if 'queue_analysis_done' not in st.session_state:
    st.session_state.queue_analysis_done = False
    
if 'gender_results' not in st.session_state:
    st.session_state.gender_results = {
        'men_count': 0,
        'women_count': 0,
        'description': "",
        'image': None
    }
    
if 'queue_results' not in st.session_state:
    st.session_state.queue_results = {
        'total_counters': 0,
        'open_counters': 0,
        'closed_counters': 0,
        'overcrowded': [],
        'suggestions': "",
        'image': None
    }

# Function to create metrics display
def display_metrics():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.gender_analysis_done:
            st.metric("Men Customers", st.session_state.gender_results['men_count'])
            st.metric("Women Customers", st.session_state.gender_results['women_count'])
        else:
            st.info("No gender analysis data available yet.")
    
    with col2:
        if st.session_state.queue_analysis_done:
            st.metric("Total Counters", st.session_state.queue_results['total_counters'])
            st.metric("Open Counters", st.session_state.queue_results['open_counters'])
            st.metric("Closed Counters", st.session_state.queue_results['closed_counters'])
        else:
            st.info("No queue analysis data available yet.")
    
    with col3:
        if st.session_state.gender_analysis_done:
            # Gender distribution pie chart
            fig = px.pie(
                values=[st.session_state.gender_results['men_count'], st.session_state.gender_results['women_count']],
                names=['Men', 'Women'],
                title='Gender Distribution',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
        elif st.session_state.queue_analysis_done:
            # Counter status bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Open', 'Closed'],
                y=[st.session_state.queue_results['open_counters'], st.session_state.queue_results['closed_counters']],
                marker_color=['green', 'red']
            ))
            fig.update_layout(title_text='Counter Status')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for visualization.")

# Function to display insights
def display_insights():
    if st.session_state.gender_analysis_done or st.session_state.queue_analysis_done:
        st.subheader("Insights")
        
        if st.session_state.gender_analysis_done:
            with st.expander("Gender Demography Insights", expanded=True):
                st.write(st.session_state.gender_results['description'])
                if st.session_state.gender_results['image'] is not None:
                    st.image(st.session_state.gender_results['image'], caption="Analyzed Image", use_column_width=True)
        
        if st.session_state.queue_analysis_done:
            with st.expander("Queue Management Insights", expanded=True):
                st.write(f"**Suggestions:** {st.session_state.queue_results['suggestions']}")
                
                if len(st.session_state.queue_results['overcrowded']) > 0:
                    st.warning(f"Overcrowded counters: {', '.join(map(str, st.session_state.queue_results['overcrowded']))}")
                
                if st.session_state.queue_results['image'] is not None:
                    st.image(st.session_state.queue_results['image'], caption="Analyzed Image", use_column_width=True)
    else:
        st.info("👆 Use the modules on the sidebar to analyze store images and generate insights.")

# Main dashboard
def main():
    st.title("🏪 Store Insights AI Dashboard")
    st.markdown("---")
    
    # Create sidebar with navigation
    st.sidebar.title("Navigation")
    
    # Display model status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Status")
    if torch.cuda.is_available():
        st.sidebar.success("✅ GPU Detected: Using MiniCPM-o model")
        st.sidebar.info(f"CUDA Version: {torch.version.cuda}")
        st.sidebar.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.warning("⚠️ No GPU Detected: Using simulated data")
        st.sidebar.info("For accurate analysis, deploy to an environment with GPU support.")
    
    # Create tabs for the dashboard
    tab1, tab2 = st.tabs(["Summary Dashboard", "Detailed Insights"])
    
    with tab1:
        st.header("Store Performance Summary")
        display_metrics()
    
    with tab2:
        display_insights()
    
    # Sidebar navigation to other pages
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Modules")
    
    if st.sidebar.button("🧑‍🧑‍🧒 Gender Demography Analysis"):
        st.switch_page("pages/gender_demographics.py")
    
    if st.sidebar.button("🧍‍♂️ Queue Management Analysis"):
        st.switch_page("pages/queue_management.py")

if __name__ == "__main__":
    main() 