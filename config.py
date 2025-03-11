"""
Configuration settings for Store Insights AI application.
This centralized configuration makes deployment easier by handling environment
variables and providing default values.
"""

import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env files
load_dotenv()  # First try default .env
load_dotenv('.env.local', override=True)  # Then try .env.local with priority

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StoreInsightsConfig")

# API Keys and External Services
DEEPINFRA_API_KEY = os.getenv('DEEPINFRA_API_KEY')
if not DEEPINFRA_API_KEY:
    logger.warning("DEEPINFRA_API_KEY not found in environment variables")
    # Don't set a default here for security reasons

# Model Configuration
LLAMA_VISION_MODEL_ID = os.getenv('LLAMA_VISION_MODEL_ID', "meta-llama/Llama-3.2-11B-Vision-Instruct")
LLAMA_VISION_MODEL_ID_90B = os.getenv('LLAMA_VISION_MODEL_ID_90B', "meta-llama/Llama-3.2-90B-Vision-Instruct")
PHI_VISION_MODEL_ID = os.getenv('PHI_VISION_MODEL_ID', "microsoft/Phi-4-multimodal-instruct")
TEXT_MODEL_ID = os.getenv('TEXT_MODEL_ID', "meta-llama/Meta-Llama-3.1-8B-Instruct")

# Set default vision model to use (phi or llama)
VISION_MODEL_ID = os.getenv('VISION_MODEL_ID', LLAMA_VISION_MODEL_ID)

MAX_TOKENS = int(os.getenv('MAX_TOKENS', 10000))

# DeepInfra API URL
DEEPINFRA_API_URL = os.getenv('DEEPINFRA_API_URL', "https://api.deepinfra.com/v1/openai")

# Application Settings
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'llama')  # 'phi', 'llama', or 'llama-90b'

# Deployment Settings
PORT = int(os.getenv('PORT', 8501))
ENABLE_CORS = os.getenv('ENABLE_CORS', 'false').lower() == 'true'

def get_api_key():
    """
    Get the DeepInfra API key with error handling.
    
    Returns:
        str: The API key if available
        
    Raises:
        ValueError: If the API key is not configured
    """
    if not DEEPINFRA_API_KEY:
        raise ValueError("DEEPINFRA_API_KEY environment variable is not set. "
                         "Please set it in your .env.local file or environment variables.")
    return DEEPINFRA_API_KEY 