#!/bin/bash
# Simple launcher script for Store Insights AI

# Parse command line arguments
USE_SMALL_MODEL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --small-model)
      USE_SMALL_MODEL=true
      shift
      ;;
    *)
      shift
      ;;
  esac
done

echo "Starting Store Insights AI..."

# Kill any existing Streamlit processes
pkill -f streamlit || true

# Wait a moment for processes to terminate
sleep 2

# Launch Streamlit with the correct server settings
if [ "$USE_SMALL_MODEL" = true ]; then
  echo "Using smaller model to avoid memory issues"
  streamlit run app/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    -- --small-model
else
  streamlit run app/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.serverAddress="0.0.0.0" \
    --browser.gatherUsageStats=false 
fi 