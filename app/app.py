# streamlit run --server.port 8501 store-inisghts-ai/app/app.py
import streamlit as st
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
import torch

# Add the app directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StoreInsightsApp")

# Fix for torch watcher error in Streamlit
try:
    # Disable Streamlit's file watcher for torch to avoid __path__._path errors
    import streamlit.watcher.path_watcher as pw
    
    orig_is_watchable = pw.is_watchable
    def patched_is_watchable(path_string):
        if 'torch' in path_string:
            return False
        return orig_is_watchable(path_string)
    
    pw.is_watchable = patched_is_watchable
    
    # Now import torch safely
    import torch
    has_gpu = torch.cuda.is_available()
    logger.info(f"CUDA available: {has_gpu}")
except Exception as e:
    logger.warning(f"Error handling torch imports: {str(e)}")
    has_gpu = False
    logger.info("CUDA not available (error during import). Will use Llama API only.")

# Import the model inference
from model.inference import get_model as get_llava_model
from model.inference_llama import get_model as get_llama_model

# Helper function to check GPU availability
def check_gpu_availability():
    """Check if GPU is available and return system GPU info."""
    gpu_info = {}
    
    try:
        has_gpu = torch.cuda.is_available()
        
        if has_gpu:
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['current_device'] = torch.cuda.current_device()
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
            logger.info(f"CUDA is available. Detected {gpu_info['device_count']} GPU(s).")
            logger.info(f"Current CUDA device: {gpu_info['current_device']}")
            logger.info(f"CUDA device name: {gpu_info['device_name']}")
        else:
            logger.info("CUDA is not available. Will use Llama API only.")
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {str(e)}")
        has_gpu = False
        logger.info("CUDA availability check failed. Will use Llama API only.")
    
    return has_gpu, gpu_info

# Check if GPU is available for Llava
has_gpu, gpu_info = check_gpu_availability()

# Check if we should use a smaller model - use configuration setting if available
parser = argparse.ArgumentParser(description='Store Insights AI')
parser.add_argument('--small-model', action='store_true', help='Use a smaller model to avoid memory issues')
args, unknown = parser.parse_known_args()
use_small_model = args.small_model or config.USE_SMALL_MODEL

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

# Create a session state variable to keep track of the selected model
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'llama'  # Default to Llama
    
if 'model' not in st.session_state:
    st.session_state.model = None

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
    </style>
    """, unsafe_allow_html=True)
    
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
            st.metric("👨 Men", "-")
        with col2:
            st.metric("👩 Women", "-")
    
    # Queue Management Metrics
    if st.session_state.queue_management_results:
        results = st.session_state.queue_management_results
        with col3:
            st.metric("🔢 Total Counters", results["total_counters"])
        with col4:
            st.metric("✅ Open Counters", results["open_counters"])
    else:
        with col3:
            st.metric("🔢 Total Counters", "-")
        with col4:
            st.metric("✅ Open Counters", "-")

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
    # Load the selected model if it hasn't been loaded yet
    if st.session_state.model is None:
        with st.spinner(f"Loading {st.session_state.selected_model.upper()} model..."):
            if st.session_state.selected_model == 'llama':
                st.session_state.model = get_llama_model()
            else:  # llava
                st.session_state.model = get_llava_model(use_small_model=use_small_model)
            
            # Log model info after it's loaded
            logger.info(f"Model loaded: {st.session_state.selected_model}")
            logger.info(f"Using mock data: {st.session_state.model.is_mock}")
    
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
        </style>
        """, unsafe_allow_html=True)
        
        # Model status indicator
        st.markdown("### Model Status")
        if st.session_state.model is not None and st.session_state.model.is_mock:
            st.markdown("<div class='model-status model-status-mock'>Using Simulated Data</div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="margin-top: 10px; font-size: 0.8em;">
            The model is using simulated data because no GPU is available or an error occurred during model loading.
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.model is not None:
            status_color = "#28a745" if not st.session_state.model.is_mock else "#dc3545"
            model_name = "Llama-3.2-90B-Vision" if st.session_state.selected_model == "llama" else "LLaVA-1.5-7B"
            
            st.markdown(f"""
            <div style="background-color: {status_color}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 10px;">
                <strong>Model Active:</strong> {model_name}
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.selected_model == "llama":
                st.markdown("""
                <div style="margin-top: 10px; font-size: 0.8em;">
                Using DeepInfra's Llama 3.2 API for vision analysis - faster but might be less accurate.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="margin-top: 10px; font-size: 0.8em;">
                Using local LLaVA model for vision analysis - may be slower but potentially more accurate.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Model loading in progress...")
        
        # Add model selection at the top of the sidebar
        st.markdown("### Model Selection")
        
        # Only show model selection if GPU is available, otherwise default to Llama
        if has_gpu:
            model_options = ['llama', 'llava']
            model_selection = st.selectbox(
                "Select Vision Model",
                options=model_options,
                index=0,  # Default to Llama
                help="Llama is faster and uses less memory. Llava may provide more accurate results but requires a GPU."
            )
            
            if model_selection != st.session_state.selected_model:
                st.session_state.selected_model = model_selection
                st.session_state.model = None  # Reset the model so it will be reloaded
                st.rerun()  # Rerun the app to load the new model
        else:
            st.info("GPU not available. Using Llama model.")
            st.session_state.selected_model = 'llama'
        
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
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
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