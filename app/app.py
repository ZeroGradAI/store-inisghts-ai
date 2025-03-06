import streamlit as st
import os
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import torch
import sys
import time
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StoreInsightsApp")

# Add the app directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model inference
from model.inference import get_model

# Check if we should use a smaller model
parser = argparse.ArgumentParser(description='Store Insights AI')
parser.add_argument('--small-model', action='store_true', help='Use a smaller model to avoid memory issues')
args, unknown = parser.parse_known_args()

# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Store Insights AI",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Get the model instance
model = get_model(use_small_model=args.small_model)

# Initialize session state for storing analysis results
if "gender_demographics_results" not in st.session_state:
    st.session_state.gender_demographics_results = None

if "queue_management_results" not in st.session_state:
    st.session_state.queue_management_results = None

# Initialize current page in session state if not present
if "current_page" not in st.session_state:
    st.session_state.current_page = "dashboard"

def display_metrics():
    """Display metrics from both analysis modules if available."""
    st.subheader("📊 Store Metrics Overview")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Gender Demographics Metrics
    if st.session_state.gender_demographics_results:
        results = st.session_state.gender_demographics_results
        with col1:
            st.metric("👨 Men", results["men_count"])
        with col2:
            st.metric("👩 Women", results["women_count"])
    else:
        with col1:
            st.metric("👨 Men", "N/A")
        with col2:
            st.metric("👩 Women", "N/A")
    
    # Queue Management Metrics
    if st.session_state.queue_management_results:
        results = st.session_state.queue_management_results
        with col3:
            st.metric("🔢 Total Counters", results["total_counters"])
        with col4:
            st.metric("✅ Open Counters", results["open_counters"])
    else:
        with col3:
            st.metric("🔢 Total Counters", "N/A")
        with col4:
            st.metric("✅ Open Counters", "N/A")

def display_insights():
    """Display insights from both analysis modules if available."""
    st.subheader("🔍 Latest Insights")
    
    col1, col2 = st.columns(2)
    
    # Gender Demographics Insights
    with col1:
        st.markdown("### 👫 Gender Demographics")
        if st.session_state.gender_demographics_results:
            results = st.session_state.gender_demographics_results
            
            # Display chart
            fig = px.pie(
                names=["Men", "Women"],
                values=[results["men_count"], results["women_count"]],
                title="Customer Gender Distribution",
                color_discrete_sequence=["#3366CC", "#FF6B6B"]
            )
            st.plotly_chart(fig)
            
            # Display insights
            st.markdown(f"**AI Insights:** {results['insights']}")
        else:
            st.info("No gender demographics data available. Run an analysis from the Gender Demographics module.")
    
    # Queue Management Insights
    with col2:
        st.markdown("### 🧍 Queue Management")
        if st.session_state.queue_management_results:
            results = st.session_state.queue_management_results
            
            # Display chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Open", "Closed"],
                y=[results["open_counters"], results["closed_counters"]],
                marker_color=["#4CAF50", "#F44336"]
            ))
            fig.update_layout(title_text="Counter Status")
            st.plotly_chart(fig)
            
            # Display insights
            st.markdown(f"**AI Recommendations:** {results['recommendations']}")
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
            background-color: #3E4154;
            color: white;
        }
        .warning-box {
            background-color: #5E5C2A;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # App title
        st.markdown("<h2 style='text-align: center; color: white;'>Store Insights AI</h2>", unsafe_allow_html=True)
        
        # Model status indicator
        st.markdown("### Model Status")
        if model.is_mock:
            st.markdown("<div class='model-status model-status-mock'>Using Simulated Data</div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='card warning-card'>
                <p>⚠️ The AI model is not available or CUDA is not detected. Using simulated data for demonstration purposes.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='model-status model-status-real'>Using Real AI Model</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='card success-card'>
                <p>✅ Using LLaVA-v1.5-7b model for image analysis.</p>
                <p>Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}</p>
            </div>
            """, unsafe_allow_html=True)
        
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

if __name__ == "__main__":
    logger.info("Starting Store Insights AI application")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Using mock data: {model.is_mock}")
    
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