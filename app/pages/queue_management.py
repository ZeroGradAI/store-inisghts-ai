import streamlit as st
import os
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import time
import torch

# Add the app directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the model inference
from model.inference import get_model

# Get the model instance
model = get_model()

def analyze_queue_management(image):
    """
    Analyze queue management using the MiniCPM-o model.
    Uses the actual model if CUDA is available, otherwise uses mock data.
    """
    # Show a progress bar to indicate processing
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)  # Faster for better UX
        progress_bar.progress(i + 1)
    
    # Use the model to analyze the image
    return model.analyze_queue_management(image)

# Set page configuration
st.set_page_config(
    page_title="Queue Management | Store Insights AI",
    page_icon="🧍‍♂️",
    layout="wide"
)

st.title("🧍‍♂️ Queue Management Analysis")
st.markdown("Upload an image of checkout counters to analyze queue status and optimize customer flow.")

# Create a sidebar with navigation
st.sidebar.title("Navigation")
if st.sidebar.button("🏠 Back to Dashboard"):
    st.switch_page("app.py")

st.sidebar.markdown("---")
st.sidebar.subheader("Other Modules")
if st.sidebar.button("🧑‍🧑‍🧒 Gender Demography Analysis"):
    st.switch_page("pages/gender_demographics.py")

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    # Image upload section
    st.subheader("Upload Checkout Image")
    uploaded_file = st.file_uploader("Choose an image of checkout counters (top view)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Analyze button
        if st.button("Analyze Queue Status"):
            with st.spinner("Analyzing image..."):
                # Use the model to analyze the image
                results = analyze_queue_management(image)
                
                # Save results to session state
                st.session_state.queue_results = results
                st.session_state.queue_analysis_done = True
                
                # Show success message
                st.success("Analysis complete! See results in the sidebar.")
                
                # Force a rerun to update the dashboard
                st.experimental_rerun()

with col2:
    # Results section
    st.subheader("Analysis Results")
    
    if st.session_state.queue_analysis_done:
        # Counter metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Counters", st.session_state.queue_results['total_counters'])
        with col2:
            st.metric("Open Counters", st.session_state.queue_results['open_counters'])
        with col3:
            st.metric("Closed Counters", st.session_state.queue_results['closed_counters'])
        
        # Counter status chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Open', 'Closed'],
            y=[st.session_state.queue_results['open_counters'], st.session_state.queue_results['closed_counters']],
            marker_color=['green', 'red']
        ))
        fig.update_layout(title_text='Counter Status')
        st.plotly_chart(fig, use_container_width=True)
        
        # Overcrowded counters
        if len(st.session_state.queue_results['overcrowded']) > 0:
            st.warning(f"Overcrowded counters: {', '.join(map(str, st.session_state.queue_results['overcrowded']))}")
        else:
            st.success("No overcrowded counters detected.")
        
        # Suggestions
        st.subheader("AI Recommendations")
        st.write(st.session_state.queue_results['suggestions'])
    else:
        st.info("Upload an image and click 'Analyze Queue Status' to see results here.")

# Additional information
st.markdown("---")
st.markdown("""
### How it works

1. Upload a top-view image of your store's checkout area
2. Our AI analyzes the image to:
   - Count the total number of checkout counters
   - Identify which counters are open and closed
   - Detect overcrowded counters
   - Provide recommendations for optimal queue management
3. Use these insights to improve customer flow and reduce wait times
""")

# Notes about model usage
st.sidebar.markdown("---")
if torch.cuda.is_available():
    st.sidebar.success("""
    **GPU Detected**: Using MiniCPM-o model for accurate counter detection and queue analysis.
    """)
else:
    st.sidebar.warning("""
    **No GPU Detected**: Using simulated data. For accurate analysis, deploy to an environment with GPU support.
    """) 