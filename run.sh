#!/bin/bash
# Simple launcher script for Store Insights AI

echo "Starting Store Insights AI..."

# Kill any existing Streamlit processes
pkill -f streamlit || true

# Wait a moment for processes to terminate
sleep 2

# Launch Streamlit with the correct server settings
streamlit run app/app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --browser.serverAddress="0.0.0.0" \
  --browser.gatherUsageStats=false 