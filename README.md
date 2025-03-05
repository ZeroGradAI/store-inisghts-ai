# Store Insights AI

A powerful AI-powered analytics tool for retail store managers to gain insights from in-store camera images.

## Features

- **Gender Demographics Analysis**: Upload images of customers browsing products to analyze gender distribution and customer behavior.
- **Queue Management Analysis**: Upload images of checkout counters to analyze queue status and optimize customer flow.
- **Dashboard**: View aggregated insights and metrics from both analysis modules.

## Technology Stack

- **Frontend**: Streamlit
- **AI Model**: MiniCPM-o 2.6B (multimodal vision-language model)
- **Data Visualization**: Plotly

## Requirements

- Python 3.8+
- CUDA-compatible GPU (for production use)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/ZeroGradAI/store-insights-ai.git
   cd store-insights-ai
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app/app.py
   ```

## Usage

1. Navigate to the application in your web browser (typically at http://localhost:8501)
2. Use the sidebar to navigate between the Dashboard, Gender Demographics, and Queue Management modules
3. Upload images in each module to analyze and generate insights
4. View the results and recommendations provided by the AI

## Application Structure

The application consists of three main pages:

- **Dashboard**: Shows an overview of all metrics and insights from both analysis modules
- **Gender Demographics**: Allows uploading and analyzing images of customers browsing products
- **Queue Management**: Allows uploading and analyzing images of checkout counters

The navigation system uses Streamlit's session state to manage page transitions, ensuring a clean single-page application experience.

## Deployment

For production use, deploy to an environment with GPU support, such as:
- Pytorch Lightning Studio
- Google Colab
- AWS EC2 with GPU
- Azure ML

### Running on Lightning Studios

To run the application on Lightning Studios:

1. Clone the repository to your Lightning Studios workspace
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make the launcher script executable:
   ```
   chmod +x launch.py
   ```
4. Run the launcher script:
   ```
   python launch.py
   ```
5. Access the application using the External URL provided by Lightning Studios
   - If the External URL shows a blank page, try accessing it with `/app` appended to the URL
   - For example: `http://18.117.245.113:8501/app`

## License

MIT

## Credits

- MiniCPM-o model by OpenBMB: https://github.com/OpenBMB/MiniCPM-o 

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

# Add the app directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model inference
from model.inference import get_model

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
model = get_model()

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
            st.plotly_chart(fig, use_container_width=True)
            
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
            st.plotly_chart(fig, use_container_width=True)
            
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
        
        # Model status indicator (small and subtle)
        if not torch.cuda.is_available():
            st.markdown("""
            <div class="warning-box">
                <p>⚠️ Using simulated data (No GPU)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Navigation buttons
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
    main() 