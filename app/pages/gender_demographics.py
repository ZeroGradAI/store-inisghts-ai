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

def show():
    """Display the Gender Demographics Analysis page."""
    st.title("👫 Gender Demographics Analysis")
    st.markdown("Upload an image of customers browsing products to analyze gender distribution and customer behavior.")
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=700)
        
        # Analyze button
        if st.button("Analyze Gender Demographics"):
            # Analyze the image
            results = analyze_gender_demographics(image)
            
            # Store results in session state
            st.session_state.gender_demographics_results = results
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Gender Distribution")
                fig = px.pie(
                    names=["Men", "Women"],
                    values=[results["men_count"], results["women_count"]],
                    title="Customer Gender Distribution",
                    color_discrete_sequence=["#3366CC", "#FF6B6B"]
                )
                st.plotly_chart(fig)
            
            with col2:
                st.subheader("Analysis Results")
                st.metric("👨 Men", results["men_count"])
                st.metric("👩 Women", results["women_count"])
                
                # Display products if available
                if "products" in results:
                    st.markdown("### Products")
                    st.markdown(results["products"])
                
                st.markdown("### AI Insights")
                st.markdown(results["insights"])
    else:
        # Display sample images
        st.markdown("### Sample Images")
        st.markdown("Don't have an image? Try one of these samples:")
        
        sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "samples", "gender")
        
        # Check if sample directory exists and has files
        if os.path.exists(sample_dir) and len(os.listdir(sample_dir)) > 0:
            sample_images = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            if sample_images:
                sample_cols = st.columns(min(3, len(sample_images)))
                for i, sample_image in enumerate(sample_images[:3]):  # Show up to 3 samples
                    with sample_cols[i]:
                        img_path = os.path.join(sample_dir, sample_image)
                        st.image(img_path, caption=f"Sample {i+1}", width=220)
                        if st.button(f"Use Sample {i+1}", key=f"sample_btn_{i}"):
                            # Load and analyze the sample image
                            image = Image.open(img_path)
                            st.image(image, caption=f"Sample {i+1}", width=700)
                            
                            # Analyze the image
                            results = analyze_gender_demographics(image)
                            
                            # Store results in session state
                            st.session_state.gender_demographics_results = results
                            
                            # Display results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Gender Distribution")
                                fig = px.pie(
                                    names=["Men", "Women"],
                                    values=[results["men_count"], results["women_count"]],
                                    title="Customer Gender Distribution",
                                    color_discrete_sequence=["#3366CC", "#FF6B6B"]
                                )
                                st.plotly_chart(fig)
                            
                            with col2:
                                st.subheader("Analysis Results")
                                st.metric("👨 Men", results["men_count"])
                                st.metric("👩 Women", results["women_count"])
                                
                                # Display products if available
                                if "products" in results:
                                    st.markdown("### Products")
                                    st.markdown(results["products"])
                                
                                st.markdown("### AI Insights")
                                st.markdown(results["insights"])
        else:
            st.info("Sample images not found. Please upload your own image.")

# Run the app if this file is run directly
if __name__ == "__main__":
    # Set page configuration - only when run directly
    st.set_page_config(
        page_title="Gender Demographics | Store Insights AI",
        page_icon="👫",
        layout="wide"
    )
    show() 