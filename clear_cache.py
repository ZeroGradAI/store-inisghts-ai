#!/usr/bin/env python
"""
Script to clear Hugging Face cache and restart the model loading process
"""

import os
import shutil
import logging
import subprocess
import sys
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CacheCleaner")

def clear_huggingface_cache():
    """Clear the Hugging Face cache directory"""
    # Get the cache directory based on platform
    if sys.platform.startswith('win'):
        # Windows path
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    else:
        # Unix path
        cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    logger.info(f"Looking for Hugging Face cache at: {cache_dir}")
    
    if os.path.exists(cache_dir):
        logger.info(f"Found Hugging Face cache directory at: {cache_dir}")
        
        # List all items in the cache
        items = os.listdir(cache_dir)
        logger.info(f"Cache contains {len(items)} items")
        
        # Ask for confirmation
        logger.info("Clearing Hugging Face cache...")
        
        try:
            # Remove the cache directory
            shutil.rmtree(cache_dir)
            logger.info("Cache directory removed successfully")
            
            # Recreate the directory
            os.makedirs(cache_dir, exist_ok=True)
            logger.info("Cache directory recreated")
            
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    else:
        logger.info(f"Hugging Face cache directory not found at: {cache_dir}")
        return False

def clear_transformers_cache():
    """Clear the transformers cache directory"""
    # Get the transformers cache directory based on platform
    if sys.platform.startswith('win'):
        # Windows path
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "transformers")
    else:
        # Unix path
        cache_dir = os.path.expanduser("~/.cache/torch/transformers")
    
    logger.info(f"Looking for transformers cache at: {cache_dir}")
    
    if os.path.exists(cache_dir):
        logger.info(f"Found transformers cache directory at: {cache_dir}")
        
        # List all items in the cache
        items = os.listdir(cache_dir)
        logger.info(f"Cache contains {len(items)} items")
        
        # Ask for confirmation
        logger.info("Clearing transformers cache...")
        
        try:
            # Remove the cache directory
            shutil.rmtree(cache_dir)
            logger.info("Cache directory removed successfully")
            
            # Recreate the directory
            os.makedirs(cache_dir, exist_ok=True)
            logger.info("Cache directory recreated")
            
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    else:
        logger.info(f"Transformers cache directory not found at: {cache_dir}")
        return False

def kill_streamlit_processes():
    """Kill any running Streamlit processes"""
    logger.info("Killing any running Streamlit processes...")
    
    try:
        # Check if we're on Windows
        if sys.platform.startswith('win'):
            # Use taskkill on Windows
            subprocess.run(["taskkill", "/F", "/IM", "streamlit.exe"], check=False)
            subprocess.run(["taskkill", "/F", "/FI", "IMAGENAME eq python.exe", "/FI", "WINDOWTITLE eq streamlit"], check=False)
            logger.info("Streamlit processes killed (Windows)")
        else:
            # Use pkill on Unix-like systems
            subprocess.run(["pkill", "-f", "streamlit"], check=False)
            logger.info("Streamlit processes killed (Unix)")
        return True
    except Exception as e:
        logger.error(f"Error killing Streamlit processes: {str(e)}")
        return False

def restart_application():
    """Restart the Streamlit application"""
    logger.info("Restarting the application...")
    
    try:
        # Get the path to the app.py file
        app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "app.py")
        logger.info(f"App path: {app_path}")
        
        # Check if the small model flag should be used
        use_small_model = "--small-model" in sys.argv
        
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
        
        # Add small model flag if needed
        if use_small_model:
            cmd.extend(["--", "--small-model"])
            logger.info("Using small model flag")
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Execute the command
        if sys.platform.startswith('win'):
            # On Windows, use shell=True to ensure the command runs properly
            process = subprocess.Popen(
                ' '.join(cmd),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
        else:
            # On Unix, we can use the command list directly
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
        
        logger.info("Application restarted")
        return True
    except Exception as e:
        logger.error(f"Error restarting application: {str(e)}")
        return False

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Clear Hugging Face cache and restart the application')
    parser.add_argument('--small-model', action='store_true', help='Use a smaller model to avoid memory issues')
    parser.add_argument('--no-restart', action='store_true', help='Do not restart the application after clearing cache')
    args = parser.parse_args()
    
    logger.info("Starting cache clearing process...")
    
    # Kill any running Streamlit processes
    kill_streamlit_processes()
    
    # Clear the Hugging Face cache
    hf_cleared = clear_huggingface_cache()
    
    # Clear the transformers cache
    tf_cleared = clear_transformers_cache()
    
    if hf_cleared or tf_cleared:
        logger.info("Cache cleared successfully")
        
        # Restart the application if not disabled
        if not args.no_restart:
            # Pass the small model flag if specified
            if args.small_model:
                sys.argv.append("--small-model")
            restart_application()
        else:
            logger.info("Application restart skipped as requested")
    else:
        logger.warning("No cache was cleared")

if __name__ == "__main__":
    main() 