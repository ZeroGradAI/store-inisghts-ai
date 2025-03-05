#!/usr/bin/env python
"""
Launcher script for Store Insights AI in Lightning Studios
This script ensures the Streamlit app is properly configured for external access
"""

import os
import subprocess
import sys
import time

def main():
    print("Starting Store Insights AI in Lightning Studios...")
    
    # Kill any existing Streamlit processes
    try:
        subprocess.run(["pkill", "-f", "streamlit"], check=False)
        print("Killed existing Streamlit processes")
    except Exception as e:
        print(f"No existing Streamlit processes to kill: {e}")
    
    # Wait a moment for processes to terminate
    time.sleep(2)
    
    # Get the absolute path to the app.py file
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "app.py")
    print(f"Launching Streamlit app from: {app_path}")
    
    # Launch Streamlit with the correct server settings
    cmd = [
        "streamlit", "run", 
        app_path,
        "--server.port=8501", 
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Execute the command
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print the output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    process.wait()
    
    # Check if the process exited with an error
    if process.returncode != 0:
        print(f"Streamlit process exited with error code: {process.returncode}")
        sys.exit(process.returncode)

if __name__ == "__main__":
    main() 