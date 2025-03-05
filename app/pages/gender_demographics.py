import streamlit as st
import os
from PIL import Image
import numpy as np
import plotly.express as px
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

def analyze_gender_demographics(image):
    """
    Analyze gender demographics using the MiniCPM-o model.
    Uses the actual model if CUDA is available, otherwise uses mock data.
    """
    # Show a progress bar to indicate processing
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)  # Faster for better UX
        progress_bar.progress(i + 1)
    
    # Use the model to analyze the image
    return model.analyze_gender_demographics(image)

# Set page configuration
st.set_page_config(
    page_title="Gender Demographics Analysis | Store Insights AI",
    page_icon="🧑‍🧑‍🧒",
    layout="wide"
)

st.title("🧑‍🧑‍🧒 Gender Demographics Analysis")
st.markdown("Upload an image of customers browsing in the store to analyze gender distribution and customer interests.")

# Create a sidebar with navigation
st.sidebar.title("Navigation")
if st.sidebar.button("🏠 Back to Dashboard"):
    st.switch_page("app.py")

st.sidebar.markdown("---")
st.sidebar.subheader("Other Modules")
if st.sidebar.button("🧍‍♂️ Queue Management Analysis"):
    st.switch_page("pages/queue_management.py")

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    # Image upload section
    st.subheader("Upload Store Image")
    uploaded_file = st.file_uploader("Choose an image of customers browsing in the store", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Analyze button
        if st.button("Analyze Customer Demographics"):
            with st.spinner("Analyzing image..."):
                # Use the model to analyze the image
                results = analyze_gender_demographics(image)
                
                # Save results to session state
                st.session_state.gender_results = results
                st.session_state.gender_analysis_done = True
                
                # Show success message
                st.success("Analysis complete! See results in the sidebar.")
                
                # Force a rerun to update the dashboard
                st.experimental_rerun()

with col2:
    # Results section
    st.subheader("Analysis Results")
    
    if st.session_state.gender_analysis_done:
        # Gender metrics
        st.metric("Men Customers", st.session_state.gender_results['men_count'])
        st.metric("Women Customers", st.session_state.gender_results['women_count'])
        
        # Gender distribution chart
        fig = px.pie(
            values=[st.session_state.gender_results['men_count'], st.session_state.gender_results['women_count']],
            names=['Men', 'Women'],
            title='Gender Distribution',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Description
        st.subheader("AI Insights")
        st.write(st.session_state.gender_results['description'])
    else:
        st.info("Upload an image and click 'Analyze Customer Demographics' to see results here.")

# Additional information
st.markdown("---")
st.markdown("""
### How it works

1. Upload an image of customers browsing in your store
2. Our AI analyzes the image to:
   - Count the number of men and women
   - Identify what products they are looking at
   - Generate insights about customer interests
3. Use these insights to improve store layout and product placement
""")

# Notes about model usage
st.sidebar.markdown("---")
if torch.cuda.is_available():
    st.sidebar.success("""
    **GPU Detected**: Using MiniCPM-o model for accurate image analysis.
    """)
else:
    st.sidebar.warning("""
    **No GPU Detected**: Using simulated data. For accurate analysis, deploy to an environment with GPU support.
    """) 