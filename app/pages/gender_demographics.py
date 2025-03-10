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

# We no longer need to import the model directly
# from model.inference import get_model

# No longer needed - we'll use the model from session state
# model = get_model()

def analyze_gender_demographics(image):
    """
    Analyze gender demographics using the selected model.
    Uses the actual model if available, otherwise uses mock data.
    """
    # Check if model is in session state
    if 'model' not in st.session_state or st.session_state.model is None:
        st.error("Model not loaded. Please return to the main page.")
        return None
    
    # Show a progress bar to indicate processing
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)  # Faster for better UX
        progress_bar.progress(i + 1)
    
    # Use the model to analyze the image
    return st.session_state.model.analyze_gender_demographics(image)

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
                st.metric("👨 Men Count", results["men_count"])
                st.metric("👩 Women Count", results["women_count"])
                
                st.markdown("### Products of Interest")
                
                # Format products list for better display
                products = results["products"]
                if isinstance(products, list):
                    # If products is a list, display each item as a bullet point
                    for product in products:
                        st.markdown(f"• {product}")
                elif isinstance(products, str):
                    # If it's a comma-separated string, split and format
                    if "," in products:
                        for product in products.split(","):
                            st.markdown(f"• {product.strip()}")
                    else:
                        st.markdown(f"• {products}")
                else:
                    # Fallback
                    st.write(products)
                
                st.markdown("### Insights")
                st.write(results["insights"])
                
                # Add debug information expander
                with st.expander("Debug Information"):
                    st.write("Raw response data:")
                    st.json(results)
                    
                    # Add section to display raw model response if available
                    if "raw_response" in results:
                        st.markdown("### Raw Model Response")
                        st.text(results["raw_response"])
                    else:
                        st.info("Raw model response not available. Check logs for more details.")
                        st.markdown("To see raw responses, check the logs or add 'raw_response' to the result dictionary.")
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
                                st.metric("👨 Men Count", results["men_count"])
                                st.metric("👩 Women Count", results["women_count"])
                                
                                st.markdown("### Products of Interest")
                                
                                # Format products list for better display
                                products = results["products"]
                                if isinstance(products, list):
                                    # If products is a list, display each item as a bullet point
                                    for product in products:
                                        st.markdown(f"• {product}")
                                elif isinstance(products, str):
                                    # If it's a comma-separated string, split and format
                                    if "," in products:
                                        for product in products.split(","):
                                            st.markdown(f"• {product.strip()}")
                                    else:
                                        st.markdown(f"• {products}")
                                else:
                                    # Fallback
                                    st.write(products)
                                
                                st.markdown("### Insights")
                                st.write(results["insights"])
                                
                                # Add debug information expander
                                with st.expander("Debug Information"):
                                    st.write("Raw response data:")
                                    st.json(results)
                                    
                                    # Add section to display raw model response if available
                                    if "raw_response" in results:
                                        st.markdown("### Raw Model Response")
                                        st.text(results["raw_response"])
                                    else:
                                        st.info("Raw model response not available. Check logs for more details.")
                                        st.markdown("To see raw responses, check the logs or add 'raw_response' to the result dictionary.")
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