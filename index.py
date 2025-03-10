import os
import subprocess
import sys

# Make sure the app directory is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

def start_app():
    """
    Start the Streamlit application with the appropriate settings for Vercel.
    """
    # Set necessary environment variables for Streamlit
    os.environ["STREAMLIT_SERVER_PORT"] = os.environ.get("PORT", "8501")
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
    os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
    
    # Start Streamlit app
    subprocess.call(["streamlit", "run", "app/app.py"])

if __name__ == "__main__":
    start_app() 