import streamlit as st
import os
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QueueManagement")

# Add the app directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from utils instead of app
from utils import update_analysis_status

@st.cache_data(ttl=300)  # Cache chart for 5 minutes
def create_counter_chart(open_counters, closed_counters):
    """Create and cache the counter status chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Open", "Closed"],
        y=[open_counters, closed_counters],
        marker_color=["#4CAF50", "#F44336"]
    ))
    fig.update_layout(
        title_text="Open vs. Closed Counters",
        height=350  # Control the chart height
    )
    return fig

def analyze_queue_management(image):
    """
    Analyze queue management using the selected model.
    Uses the actual model if available, otherwise uses mock data.
    """
    # Check if model is in session state
    if 'model' not in st.session_state or st.session_state.model is None:
        st.error("Model not loaded. Please return to the main page.")
        return None
    
    try:
        # Update status to in progress
        update_analysis_status(in_progress=True)
        
        # Use the model to analyze the image
        results = st.session_state.model.analyze_queue_management(image)
        
        # Update status to complete
        update_analysis_status(in_progress=False)
        
        return results
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        update_analysis_status(in_progress=False)
        return None

def format_recommendations(recommendations):
    """Format recommendations for display."""
    if isinstance(recommendations, list):
        return recommendations
    elif isinstance(recommendations, str):
        if "," in recommendations:
            return [rec.strip() for rec in recommendations.split(",")]
        return [recommendations]
    return [str(recommendations)]

def show():
    """Display the Queue Management Analysis page."""
    st.title("🧍 Queue Management Analysis")
    st.markdown("Upload an image of checkout counters to analyze queue status and optimize customer flow.")
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Create placeholder for results
    results_placeholder = st.empty()
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=700)
        
        # Analyze button
        if st.button("Analyze Queue Management"):
            with st.spinner("Analyzing image..."):
                # Analyze the image
                results = analyze_queue_management(image)
            
            if results:
                # Store results in session state
                st.session_state.queue_management_results = results
                
                # Display results using the placeholder
                with results_placeholder.container():
                    # Section 1: Counter Information
                    st.subheader("Counter Status")
                    counter_col1, counter_col2 = st.columns([3, 2])
                    
                    with counter_col1:
                        # Use cached chart
                        fig = create_counter_chart(results["open_counters"], results["closed_counters"])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with counter_col2:
                        # Counter metrics
                        st.metric("🔢 Total Counters", results["total_counters"], delta=None)
                        st.metric("✅ Open Counters", results["open_counters"], delta=None)
                        st.metric("❌ Closed Counters", results["closed_counters"], delta=None)
                        st.metric("🧍 Customers in Queue", results["customers_in_queue"], delta=None)
                    
                    # Section 2: Wait Time & Status
                    st.markdown("---")
                    st.subheader("Queue Analysis")
                    
                    status_col1, status_col2 = st.columns(2)
                    
                    with status_col1:
                        with st.expander("Wait Time & Efficiency", expanded=True):
                            if results["customers_in_queue"] > 0:
                                if "avg_wait_time" in results and results["avg_wait_time"] not in ["Not specified", "Not enough data"]:
                                    st.info(f"⏱️ Estimated Wait Time: {results['avg_wait_time']}")
                            else:
                                st.success("✅ No waiting time - counters are clear!")
                            
                            if "queue_efficiency" in results:
                                efficiency = results["queue_efficiency"]
                                if efficiency.lower() in ["high", "good", "excellent"]:
                                    st.success(f"🎯 Queue Efficiency: {efficiency}")
                                elif efficiency.lower() in ["moderate", "average"]:
                                    st.warning(f"📊 Queue Efficiency: {efficiency}")
                                else:
                                    st.error(f"⚠️ Queue Efficiency: {efficiency}")
                    
                    with status_col2:
                        with st.expander("AI Recommendations", expanded=True):
                            recommendations = format_recommendations(results["recommendations"])
                            for rec in recommendations:
                                st.markdown(f"• {rec}")
                    
                    # Debug Information
                    with st.expander("Debug Information", expanded=False):
                        st.json({
                            "Model Type": st.session_state.selected_model,
                            "Temperature": st.session_state.temperature,
                            "Is Mock Data": results.get("is_mock", False),
                            "Raw Response": results.get("raw_response", "Not available")
                        })
                
                # Show success message
                st.success("Analysis complete! You can view the results in the dashboard.")
            else:
                st.error("Analysis failed. Please try again.")
    else:
        # Show placeholder message when no image is uploaded
        with results_placeholder.container():
            st.info("Upload an image to begin analysis.")

# Run the app if this file is run directly
if __name__ == "__main__":
    import streamlit.web.bootstrap
    if streamlit.web.bootstrap.is_running_with_streamlit:
        # Set page configuration - only when run directly
        st.set_page_config(
            page_title="Queue Management | Store Insights AI",
            page_icon="🧍",
            layout="wide"
        )
    show() 