# streamlit run --server.port 8501 store-inisghts-ai/app/app.py
import streamlit as st

# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Store Insights AI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏪"
)

import os
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import sys
import time
import logging
import argparse

# Add the app directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration and utilities
import config
from utils import update_analysis_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StoreInsightsApp")

# Import the model inference
from model.inference_llama import get_api_model

# Hide the default Streamlit menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}
[data-testid="stSidebarNav"] {display: none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Create a session state variable to keep track of the selected model
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = config.DEFAULT_MODEL  # Default to the model specified in config
    
if 'model' not in st.session_state:
    st.session_state.model = None

# Initialize temperature in session state
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.1  # Default to low temperature for more deterministic outputs

# Initialize session state for storing analysis results
if "gender_demographics_results" not in st.session_state:
    st.session_state.gender_demographics_results = None

if "queue_management_results" not in st.session_state:
    st.session_state.queue_management_results = None

# Initialize current page in session state if not present
if "current_page" not in st.session_state:
    st.session_state.current_page = "dashboard"

# Add caching for analysis results
if 'analysis_in_progress' not in st.session_state:
    st.session_state.analysis_in_progress = False

if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = None

def display_metrics():
    """Display metrics from both analysis modules if available."""
    st.subheader("📊 Store Metrics Overview")
    
    # Add CSS for centering metric values and labels
    st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        text-align: center;
        justify-content: center;
    }
    [data-testid="stMetricLabel"] {
        text-align: center;
        justify-content: center;
    }
    div[data-testid="stMetricValue"] > div {
        width: 100%;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] > div {
        width: 100%;
        text-align: center;
    }
    .analysis-status {
        padding: 5px 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
    .analysis-in-progress {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    .analysis-complete {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Show analysis status if needed
    if st.session_state.analysis_in_progress:
        st.markdown(
            '<div class="analysis-status analysis-in-progress">⏳ Analysis in progress...</div>',
            unsafe_allow_html=True
        )
    elif st.session_state.last_analysis_time:
        st.markdown(
            f'<div class="analysis-status analysis-complete">✅ Last analysis completed at {st.session_state.last_analysis_time}</div>',
            unsafe_allow_html=True
        )
    
    # Create columns for metrics with placeholder containers
    col1, col2, col3, col4 = st.columns(4)
    
    # Gender Demographics Metrics
    if st.session_state.gender_demographics_results:
        results = st.session_state.gender_demographics_results
        with col1:
            st.metric("👨 Men", results["men_count"], delta=None)
        with col2:
            st.metric("👩 Women", results["women_count"], delta=None)
    else:
        with col1:
            st.metric("👨 Men", "-", delta=None)
        with col2:
            st.metric("👩 Women", "-", delta=None)
    
    # Queue Management Metrics
    if st.session_state.queue_management_results:
        results = st.session_state.queue_management_results
        with col3:
            st.metric("🔢 Total Counters", results["total_counters"], delta=None)
        with col4:
            st.metric("✅ Open Counters", results["open_counters"], delta=None)
    else:
        with col3:
            st.metric("🔢 Total Counters", "-", delta=None)
        with col4:
            st.metric("✅ Open Counters", "-", delta=None)

@st.cache_data(ttl=300)  # Cache charts for 5 minutes
def create_gender_chart(men_count, women_count):
    """Create and cache the gender distribution chart."""
    fig = px.pie(
        names=["Men", "Women"],
        values=[men_count, women_count],
        title="Customer Gender Distribution",
        color_discrete_sequence=["#3366CC", "#FF6B6B"]
    )
    return fig

@st.cache_data(ttl=300)  # Cache charts for 5 minutes
def create_queue_chart(open_counters, closed_counters):
    """Create and cache the queue management chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Open", "Closed"],
        y=[open_counters, closed_counters],
        marker_color=["#4CAF50", "#F44336"]
    ))
    fig.update_layout(title_text="Counter Status")
    return fig

def display_insights():
    """Display insights from both analysis modules if available."""
    st.subheader("🔍 Latest Insights")
    
    col1, col2 = st.columns(2)
    
    # Gender Demographics Insights
    with col1:
        st.markdown("### 👫 Gender Demographics")
        if st.session_state.gender_demographics_results:
            results = st.session_state.gender_demographics_results
            
            # Use cached chart
            fig = create_gender_chart(results["men_count"], results["women_count"])
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("View AI Insights", expanded=True):
                st.markdown(f"{results['insights']}")
        else:
            st.info("No gender demographics data available. Run an analysis from the Gender Demographics module.")
    
    # Queue Management Insights
    with col2:
        st.markdown("### 🧍 Queue Management")
        if st.session_state.queue_management_results:
            results = st.session_state.queue_management_results
            
            # Use cached chart
            fig = create_queue_chart(results["open_counters"], results["closed_counters"])
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("View AI Recommendations", expanded=True):
                if isinstance(results['recommendations'], list):
                    for rec in results['recommendations']:
                        st.markdown(f"• {rec}")
                else:
                    st.markdown(f"{results['recommendations']}")
        else:
            st.info("No queue management data available. Run an analysis from the Queue Management module.")

def show_dashboard():
    """Display the main dashboard."""
    st.title("🏪 Store Insights AI")
    st.markdown("""
    Welcome to Store Insights AI - your intelligent assistant for retail store management.
    Upload images from your store cameras to get real-time insights on customer demographics and queue management.
    """)
    
    # Display metrics and insights if data is available
    display_metrics()
    display_insights()

def main():
    """Main function to run the Streamlit app."""
    # Load the selected model if it hasn't been loaded yet
    if st.session_state.model is None:
        with st.spinner(f"Loading {st.session_state.selected_model.upper()} model..."):
            if st.session_state.selected_model == 'phi':
                st.session_state.model = get_api_model(model_type='phi', temperature=st.session_state.temperature)
            elif st.session_state.selected_model == 'llama':
                st.session_state.model = get_api_model(model_type='llama', temperature=st.session_state.temperature)
            elif st.session_state.selected_model == 'llama-90b':
                st.session_state.model = get_api_model(model_type='llama-90b', temperature=st.session_state.temperature)
            
            # Log model info after it's loaded
            logger.info(f"Model loaded: {st.session_state.selected_model}")
            logger.info(f"Using mock data: {st.session_state.model.is_mock}")
            logger.info(f"Temperature: {st.session_state.temperature}")
    
    # Clean sidebar with styled navigation buttons
    with st.sidebar:
        # Add custom CSS for styling the buttons
        st.markdown("""
        <style>
        div.stButton > button {
            background-color: #2E303E;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 15px;
            text-align: left;
            width: 100%;
            margin-bottom: 10px;
            font-weight: 500;
        }
        div.stButton > button:hover {
            background-color: #4A4D5E;
        }
        .temperature-info {
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Model status indicator
        st.markdown("### Model Status")
        if st.session_state.model is not None and st.session_state.model.is_mock:
            st.markdown("<div class='model-status model-status-mock'>Using Simulated Data</div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="margin-top: 10px; font-size: 0.8em;">
            The model is using simulated data because an error occurred during model loading.
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.model is not None:
            status_color = "#28a745" if not st.session_state.model.is_mock else "#dc3545"
            
            # Get the model name based on the selected model
            if st.session_state.selected_model == 'phi':
                model_name = "Phi-4-multimodal-instruct"
                model_description = "Microsoft Phi-4"
            elif st.session_state.selected_model == 'llama':
                model_name = "Llama-3.2-11B-Vision-Instruct"
                model_description = "Meta Llama-3.2-11B"
            elif st.session_state.selected_model == 'llama-90b':
                model_name = "Llama-3.2-90B-Vision-Instruct"
                model_description = "Meta Llama-3.2-90B"
            
            st.markdown(f"<div class='model-status' style='background-color: {status_color};'>Model Ready: {model_name}</div>", unsafe_allow_html=True)
            
            # Add model description
            st.markdown(f"""
            <div style="margin-top: 10px; font-size: 0.8em;">
            Using DeepInfra's {model_description} API for vision analysis.
            </div>
            """, unsafe_allow_html=True)
        
        # Define available models
        model_options = ['llama', 'llama-90b', 'phi']
        model_descriptions = {
            'llama': 'meta-llama/Llama-3.2-11B-Vision-Instruct (Default)',
            'llama-90b': 'meta-llama/Llama-3.2-90B-Vision-Instruct',
            'phi': 'microsoft/Phi-4-multimodal-instruct'
        }
        
        # Format the options with descriptions
        formatted_options = [f"{model_descriptions[opt]}" for opt in model_options]
        
        # Find the index of the current model in the options
        try:
            current_index = model_options.index(st.session_state.selected_model)
        except ValueError:
            current_index = 0  # Default to first option if not found
        
        # Create the selectbox
        selection = st.selectbox(
            "Select Vision Model",
            options=formatted_options,
            index=current_index,
            help="Choose which vision model to use for image analysis"
        )
        
        # Extract the selected model type from the formatted selection
        for model_type, description in model_descriptions.items():
            if description in selection:
                selected_model = model_type
                break
        
        # If model changed, update session state and reload
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.session_state.model = None  # Reset the model so it will be reloaded
            st.rerun()  # Rerun the app to load the new model
        
        # Add temperature control after model selection
        st.markdown("### Model Settings")
        new_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Controls randomness in model outputs. Lower values (closer to 0) make the output more deterministic and focused, while higher values make it more creative and varied."
        )
        
        st.markdown("""
        <div class="temperature-info">
        🎯 Lower temperature (0.0-0.3): More deterministic, better for analysis<br>
        🎨 Higher temperature (0.7-1.0): More creative, varied outputs
        </div>
        """, unsafe_allow_html=True)
        
        # If temperature changed, update session state and reload model
        if new_temperature != st.session_state.temperature:
            st.session_state.temperature = new_temperature
            st.session_state.model = None  # Reset the model so it will be reloaded
            st.rerun()  # Rerun the app to load the new model
        
        st.markdown("---")
        
        # Navigation buttons
        st.markdown("### Navigation")
        
        if st.button("📊 Dashboard", key="dashboard_btn"):
            st.session_state.current_page = "dashboard"
            st.rerun()
        
        if st.button("👫 Gender Demographics", key="gender_btn"):
            st.session_state.current_page = "gender_demographics"
            st.rerun()
        
        if st.button("🧍 Queue Management", key="queue_btn"):
            st.session_state.current_page = "queue_management"
            st.rerun()
    
    # Display the appropriate page based on current_page
    if st.session_state.current_page == "dashboard":
        show_dashboard()
    elif st.session_state.current_page == "gender_demographics":
        from pages import gender_demographics
        gender_demographics.show()
    elif st.session_state.current_page == "queue_management":
        from pages import queue_management
        queue_management.show()

def update_analysis_status(in_progress=False):
    """Update the analysis status in session state."""
    st.session_state.analysis_in_progress = in_progress
    if not in_progress:
        st.session_state.last_analysis_time = time.strftime("%I:%M:%S %p")

if __name__ == "__main__":
    logger.info("Starting Store Insights AI application")
    
    main() 

# Add this section to the end of the file to properly configure Streamlit for external access
# This will be used when running on Lightning Studios or other cloud environments
if os.environ.get('LIGHTNING_APP_STATE_URL'):
    # We're running in Lightning Studios
    import subprocess
    import sys
    
    # Kill any existing Streamlit processes
    subprocess.run(["pkill", "-f", "streamlit"])
    
    # Launch Streamlit with the correct server settings
    subprocess.run([
        "streamlit", "run", 
        os.path.abspath(__file__),
        "--server.port=8501", 
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ]) 