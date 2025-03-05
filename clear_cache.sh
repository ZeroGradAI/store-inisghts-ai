#!/bin/bash
# Script to clear Hugging Face cache and restart the application

echo "Starting cache clearing process..."

# Kill any running Streamlit processes
echo "Killing any running Streamlit processes..."
pkill -f streamlit || true
echo "Streamlit processes killed"

# Clear the Hugging Face cache
echo "Clearing Hugging Face cache..."
HF_CACHE_DIR=~/.cache/huggingface
if [ -d "$HF_CACHE_DIR" ]; then
    echo "Found Hugging Face cache directory at: $HF_CACHE_DIR"
    echo "Cache contains $(ls -1 $HF_CACHE_DIR | wc -l) items"
    rm -rf $HF_CACHE_DIR
    echo "Cache directory removed successfully"
    mkdir -p $HF_CACHE_DIR
    echo "Cache directory recreated"
    HF_CLEARED=true
else
    echo "Hugging Face cache directory not found at: $HF_CACHE_DIR"
    HF_CLEARED=false
fi

# Clear the transformers cache
echo "Clearing transformers cache..."
TF_CACHE_DIR=~/.cache/torch/transformers
if [ -d "$TF_CACHE_DIR" ]; then
    echo "Found transformers cache directory at: $TF_CACHE_DIR"
    echo "Cache contains $(ls -1 $TF_CACHE_DIR | wc -l) items"
    rm -rf $TF_CACHE_DIR
    echo "Cache directory removed successfully"
    mkdir -p $TF_CACHE_DIR
    echo "Cache directory recreated"
    TF_CLEARED=true
else
    echo "Transformers cache directory not found at: $TF_CACHE_DIR"
    TF_CLEARED=false
fi

# Restart the application
if [ "$HF_CLEARED" = true ] || [ "$TF_CLEARED" = true ]; then
    echo "Cache cleared successfully"
    echo "Restarting the application..."
    
    # Get the path to the app.py file
    APP_PATH="$(dirname "$(realpath "$0")")/app/app.py"
    
    # Launch Streamlit with the correct server settings
    streamlit run $APP_PATH \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false &
    
    echo "Application restarted"
else
    echo "No cache was cleared"
fi 