"""
Test script for the Microsoft Phi-4 model integration.
This verifies that the Phi-4 model is correctly configured and can analyze images.
"""

from PIL import Image
import os
import logging
import sys
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PhiModelTest")

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from model.inference_llama import APIModelInference, get_api_model

def print_separator():
    """Print a separator line for better readability."""
    print("\n" + "="*80 + "\n")

def test_phi_model_directly():
    """Test the Phi model directly through the APIModelInference class."""
    print_separator()
    logger.info("Testing Microsoft Phi-4 model directly")
    
    # Initialize the model
    model = APIModelInference(model_type='phi')
    
    # Verify model is properly initialized
    logger.info(f"Model initialized with vision model ID: {model.vision_model_id}")
    logger.info(f"Model name: {model.get_model_name()}")
    
    # Check if model is using mock data
    if model.is_mock:
        logger.warning("Model is using mock data. API key may be missing or invalid.")
        return
    
    # Path to test image
    image_path = "samples/gender/store_image.jpg"
    if not os.path.exists(image_path):
        logger.error(f"Test image not found: {image_path}")
        return
    
    # Load and analyze image
    image = Image.open(image_path)
    logger.info("Analyzing gender demographics with Phi model...")
    
    # Analyze gender demographics
    results = model.analyze_gender_demographics(image)
    
    # Print results
    logger.info("Gender demographics results:")
    logger.info(f"Men count: {results.get('men_count', 'Not available')}")
    logger.info(f"Women count: {results.get('women_count', 'Not available')}")
    logger.info(f"Products: {results.get('products', 'Not available')}")
    
    print_separator()
    logger.info("Testing queue management analysis with Phi model...")
    
    # Path to queue image
    image_path = "samples/queue/store_image.jpg"
    if not os.path.exists(image_path):
        logger.error(f"Test image not found: {image_path}")
        return
    
    # Load and analyze image
    image = Image.open(image_path)
    
    # Analyze queue management
    results = model.analyze_queue_management(image)
    
    # Print results
    logger.info("Queue management results:")
    logger.info(f"Open counters: {results.get('open_counters', 'Not available')}")
    logger.info(f"Closed counters: {results.get('closed_counters', 'Not available')}")
    logger.info(f"Total counters: {results.get('total_counters', 'Not available')}")
    logger.info(f"Customers in queue: {results.get('customers_in_queue', 'Not available')}")

def test_get_api_model_function():
    """Test the get_api_model function that the app will use."""
    print_separator()
    logger.info("Testing get_api_model function with default (phi) model")
    
    # Get default model (should be phi)
    model = get_api_model()
    logger.info(f"Default model type: {model.model_type}")
    logger.info(f"Model name: {model.get_model_name()}")
    
    print_separator()
    logger.info("Testing get_api_model function with explicit phi model")
    
    # Get explicit phi model
    phi_model = get_api_model(model_type='phi')
    logger.info(f"Requested phi model type: {phi_model.model_type}")
    logger.info(f"Model name: {phi_model.get_model_name()}")
    
    print_separator()
    logger.info("Testing get_api_model function with llama model")
    
    # Get llama model
    llama_model = get_api_model(model_type='llama')
    logger.info(f"Requested llama model type: {llama_model.model_type}")
    logger.info(f"Model name: {llama_model.get_model_name()}")

if __name__ == "__main__":
    load_dotenv('.env.local')  # Load environment variables
    
    print_separator()
    logger.info("STARTING PHI MODEL INTEGRATION TESTS")
    print_separator()
    
    # Test the get_api_model function
    test_get_api_model_function()
    
    # Test direct model usage
    test_phi_model_directly()
    
    print_separator()
    logger.info("PHI MODEL INTEGRATION TESTS COMPLETED")
    print_separator() 