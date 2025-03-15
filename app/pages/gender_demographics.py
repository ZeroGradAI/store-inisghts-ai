import streamlit as st
import os
from PIL import Image
import numpy as np
import plotly.express as px
import sys
import time

# Add the app directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from utils instead of app
from utils import update_analysis_status

@st.cache_data(ttl=300)  # Cache chart for 5 minutes
def create_gender_chart(men_count, women_count):
    """Create and cache the gender distribution chart."""
    fig = px.pie(
        names=["Men", "Women"],
        values=[men_count, women_count],
        title="Customer Gender Distribution",
        color_discrete_sequence=["#3366CC", "#FF6B6B"]
    )
    return fig

def analyze_gender_demographics(image):
    """
    Analyze gender demographics using the selected model.
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
        results = st.session_state.model.analyze_gender_demographics(image)
        
        # Update status to complete
        update_analysis_status(in_progress=False)
        
        return results
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        update_analysis_status(in_progress=False)
        return None

def format_products(products):
    """Format products list for display."""
    if isinstance(products, list):
        return [f"• {product}" for product in products]
    elif isinstance(products, str):
        if "," in products:
            return [f"• {product.strip()}" for product in products.split(",")]
        return [f"• {products}"]
    return [f"• {str(products)}"]

def show():
    """Display the Gender Demographics Analysis page."""
    st.title("👫 Gender Demographics Analysis")
    st.markdown("Upload an image of customers browsing products to analyze gender distribution and customer behavior.")
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Create placeholder for results
    results_placeholder = st.empty()
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=700)
        
        # Analyze button
        if st.button("Analyze Gender Demographics"):
            with st.spinner("Analyzing image..."):
                # Analyze the image
                results = analyze_gender_demographics(image)
            
            if results:
                # Store results in session state
                st.session_state.gender_demographics_results = results
                
                # Display results using the placeholder
                with results_placeholder.container():
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Gender Distribution")
                        # Use cached chart
                        fig = create_gender_chart(results["men_count"], results["women_count"])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Analysis Results")
                        st.metric("👨 Men Count", results["men_count"], delta=None)
                        st.metric("👩 Women Count", results["women_count"], delta=None)
                        
                        with st.expander("Products of Interest", expanded=True):
                            for product_line in format_products(results["products"]):
                                st.markdown(product_line)
                        
                        with st.expander("AI Insights", expanded=True):
                            st.write(results["insights"])
                        
                        # Add debug information expander
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
            page_title="Gender Demographics | Store Insights AI",
            page_icon="👫",
            layout="wide"
        )
    show() 