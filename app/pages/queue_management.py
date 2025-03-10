import streamlit as st
import os
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import time
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QueueManagement")

# Add the app directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# We no longer need to import the model directly
# from model.inference import get_model

# No longer needed - we'll use the model from session state
# model = get_model()

def analyze_queue_management(image):
    """
    Analyze queue management using the selected model.
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
    return st.session_state.model.analyze_queue_management(image)

def show():
    """Display the Queue Management Analysis page."""
    st.title("🧍 Queue Management Analysis")
    st.markdown("Upload an image of checkout counters to analyze queue status and optimize customer flow.")
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=700)
        
        # Analyze button
        if st.button("Analyze Queue Management"):
            # Analyze the image
            results = analyze_queue_management(image)
            
            # Store results in session state
            st.session_state.queue_management_results = results
            
            # Organize layout with clear divisions
            # Section 1: Counter Information (chart side by side with metrics)
            st.subheader("Counter Status")
            counter_col1, counter_col2 = st.columns([3, 2])
            
            with counter_col1:
                # Counter status chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["Open", "Closed"],
                    y=[results["open_counters"], results["closed_counters"]],
                    marker_color=["#4CAF50", "#F44336"]
                ))
                fig.update_layout(
                    title_text="Open vs. Closed Counters",
                    height=350  # Control the chart height
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with counter_col2:
                # Counter metrics
                st.metric("🔢 Total Counters", results["total_counters"])
                st.metric("✅ Open Counters", results["open_counters"])
                st.metric("❌ Closed Counters", results["closed_counters"])
                st.metric("🧍 Customers in Queue", results["customers_in_queue"])
            
            # Section 2: Wait Time & Status in its own section with clear separation
            st.markdown("---")
            st.subheader("Wait Time & Status")
            
            # Put status info in a container with custom styling
            status_container = st.container()
            with status_container:
                # Only show wait time if there are customers in queue
                if results["customers_in_queue"] > 0:
                    if "avg_wait_time" in results and results["avg_wait_time"] not in ["Not specified", "Not enough data"]:
                        st.info(f"⏱️ Average Wait Time: **{results['avg_wait_time']}**")
                    else:
                        st.info("⏱️ Average Wait Time: **Not available**")
                else:
                    st.info("⏱️ No customers in queue - no wait time")
                
                # Display overcrowded status with clearer formatting
                if "overcrowded_counters" in results:
                    if results["overcrowded_counters"] == True:
                        st.warning("⚠️ **Checkout counters are overcrowded!**")
                    elif results["customers_in_queue"] > 0:
                        st.success("✅ **Queue management is efficient**")
                    else:
                        st.success("✅ **No waiting customers**")
            
            # Section 3: Recommendations
            st.markdown("---")
            st.subheader("AI Recommendations")
            
            # Format recommendations for better display
            recommendations = results["recommendations"]
            if isinstance(recommendations, list):
                # If recommendations is a list, display each item as a bullet point
                for recommendation in recommendations:
                    st.markdown(f"• {recommendation}")
            elif isinstance(recommendations, str):
                # If it's a comma-separated string, split and format
                if "," in recommendations:
                    for recommendation in recommendations.split(","):
                        st.markdown(f"• {recommendation.strip()}")
                else:
                    st.markdown(recommendations)
            else:
                # Fallback
                st.write(recommendations)
            
            # Debug information (in collapsed section for developers)
            with st.expander("Debug Information"):
                st.write("Raw response data:")
                st.json(results)
                
                # Add section to display raw model response if available
                if "raw_response" in results:
                    st.markdown("### Raw Model Response")
                    st.text(results["raw_response"])
                else:
                    st.info("Raw model response not available. Check logs for more details.")
                    st.markdown("To see raw responses, check the logs or add 'raw_response' to the result dictionary in model/inference.py.")
    else:
        # Display sample images
        st.markdown("### Sample Images")
        st.markdown("Don't have an image? Try one of these samples:")
        
        sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "samples", "queue")
        
        # Check if sample directory exists and has files
        if os.path.exists(sample_dir) and len(os.listdir(sample_dir)) > 0:
            sample_images = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            if sample_images:
                sample_cols = st.columns(min(3, len(sample_images)))
                for i, sample_image in enumerate(sample_images[:3]):  # Show up to 3 samples
                    with sample_cols[i]:
                        img_path = os.path.join(sample_dir, sample_image)
                        st.image(img_path, caption=f"Sample {i+1}", width=220)
                        if st.button(f"Use Sample {i+1}", key=f"sample_{i}"):
                            # Load and analyze the sample image
                            image = Image.open(img_path)
                            st.image(image, caption=f"Sample {i+1}", width=700)
                            
                            # Analyze the image
                            results = analyze_queue_management(image)
                            
                            # Store results in session state
                            st.session_state.queue_management_results = results
                            
                            # Organize layout with clear divisions
                            # Section 1: Counter Information (chart side by side with metrics)
                            st.subheader("Counter Status")
                            counter_col1, counter_col2 = st.columns([3, 2])
                            
                            with counter_col1:
                                # Counter status chart
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=["Open", "Closed"],
                                    y=[results["open_counters"], results["closed_counters"]],
                                    marker_color=["#4CAF50", "#F44336"]
                                ))
                                fig.update_layout(
                                    title_text="Open vs. Closed Counters",
                                    height=350  # Control the chart height
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with counter_col2:
                                # Counter metrics
                                st.metric("🔢 Total Counters", results["total_counters"])
                                st.metric("✅ Open Counters", results["open_counters"])
                                st.metric("❌ Closed Counters", results["closed_counters"])
                                st.metric("🧍 Customers in Queue", results["customers_in_queue"])
                            
                            # Section 2: Wait Time & Status in its own section with clear separation
                            st.markdown("---")
                            st.subheader("Wait Time & Status")
                            
                            # Put status info in a container with custom styling
                            status_container = st.container()
                            with status_container:
                                # Only show wait time if there are customers in queue
                                if results["customers_in_queue"] > 0:
                                    if "avg_wait_time" in results and results["avg_wait_time"] not in ["Not specified", "Not enough data"]:
                                        st.info(f"⏱️ Average Wait Time: **{results['avg_wait_time']}**")
                                    else:
                                        st.info("⏱️ Average Wait Time: **Not available**")
                                else:
                                    st.info("⏱️ No customers in queue - no wait time")
                                
                                # Display overcrowded status with clearer formatting
                                if "overcrowded_counters" in results:
                                    if results["overcrowded_counters"] == True:
                                        st.warning("⚠️ **Checkout counters are overcrowded!**")
                                    elif results["customers_in_queue"] > 0:
                                        st.success("✅ **Queue management is efficient**")
                                    else:
                                        st.success("✅ **No waiting customers**")
                            
                            # Section 3: Recommendations
                            st.markdown("---")
                            st.subheader("AI Recommendations")
                            
                            # Format recommendations for better display
                            recommendations = results["recommendations"]
                            if isinstance(recommendations, list):
                                # If recommendations is a list, display each item as a bullet point
                                for recommendation in recommendations:
                                    st.markdown(f"• {recommendation}")
                            elif isinstance(recommendations, str):
                                # If it's a comma-separated string, split and format
                                if "," in recommendations:
                                    for recommendation in recommendations.split(","):
                                        st.markdown(f"• {recommendation.strip()}")
                                else:
                                    st.markdown(recommendations)
                            else:
                                # Fallback
                                st.write(recommendations)
                            
                            # Debug information (in collapsed section for developers)
                            with st.expander("Debug Information"):
                                st.write("Raw response data:")
                                st.json(results)
                                
                                # Add section to display raw model response if available
                                if "raw_response" in results:
                                    st.markdown("### Raw Model Response")
                                    st.text(results["raw_response"])
                                else:
                                    st.info("Raw model response not available. Check logs for more details.")
                                    st.markdown("To see raw responses, check the logs or add 'raw_response' to the result dictionary in model/inference.py.")
        else:
            st.info("Sample images not found. Please upload your own image.")

# Run the app if this file is run directly
if __name__ == "__main__":
    # Set page configuration - only when run directly
    st.set_page_config(
        page_title="Queue Management | Store Insights AI",
        page_icon="🧍",
        layout="wide"
    )
    show() 