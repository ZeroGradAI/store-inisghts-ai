from PIL import Image
import os
import logging
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.inference_llama import LlamaModelInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_gender_demographics():
    """Test the gender demographics analysis using the two-step approach."""
    # Initialize the model
    model = LlamaModelInference()
    
    # Path to test image
    image_path = "samples/gender/store_image.jpg"
    
    # Check if image exists
    if not os.path.exists(image_path):
        logger.error(f"Test image not found: {image_path}")
        return
    
    # Load image
    image = Image.open(image_path)
    
    # Analyze image
    logger.info("Starting gender demographics analysis...")
    results = model.analyze_gender_demographics(image)
    
    # Print results
    logger.info("Gender Demographics Results:")
    logger.info(f"Men Count: {results.get('men_count', 'Not available')}")
    logger.info(f"Women Count: {results.get('women_count', 'Not available')}")
    logger.info(f"Products: {results.get('products', 'Not available')}")
    logger.info(f"Insights: {results.get('insights', 'Not available')}")
    logger.info(f"Is Mock Data: {results.get('is_mock', True)}")
    
    # Print raw responses
    logger.info("\nVision Model Raw Response:")
    logger.info(results.get('vision_raw_response', 'Not available')[:500] + '...')
    
    logger.info("\nText Model Raw Response (JSON):")
    logger.info(results.get('text_raw_response', 'Not available'))
    
    return results

def test_queue_management():
    """Test the queue management analysis using the two-step approach."""
    # Initialize the model
    model = LlamaModelInference()
    
    # Path to test image
    image_path = "samples/queue/store_image.jpg"
    
    # Check if image exists
    if not os.path.exists(image_path):
        logger.error(f"Test image not found: {image_path}")
        return
    
    # Load image
    image = Image.open(image_path)
    
    # Analyze image
    logger.info("Starting queue management analysis...")
    results = model.analyze_queue_management(image)
    
    # Print results
    logger.info("Queue Management Results:")
    logger.info(f"Open Counters: {results.get('open_counters', 'Not available')}")
    logger.info(f"Closed Counters: {results.get('closed_counters', 'Not available')}")
    logger.info(f"Total Counters: {results.get('total_counters', 'Not available')}")
    logger.info(f"Customers in Queue: {results.get('customers_in_queue', 'Not available')}")
    logger.info(f"Average Wait Time: {results.get('avg_wait_time', 'Not available')}")
    logger.info(f"Queue Efficiency: {results.get('queue_efficiency', 'Not available')}")
    logger.info(f"Overcrowded Counters: {results.get('overcrowded_counters', 'Not available')}")
    logger.info(f"Is Mock Data: {results.get('is_mock', True)}")
    
    # Print raw responses
    logger.info("\nVision Model Raw Response:")
    logger.info(results.get('vision_raw_response', 'Not available')[:500] + '...')
    
    logger.info("\nText Model Raw Response (JSON):")
    logger.info(results.get('text_raw_response', 'Not available'))
    
    return results

if __name__ == "__main__":
    # Test both analyses
    logger.info("===== TESTING GENDER DEMOGRAPHICS ANALYSIS =====")
    gender_results = test_gender_demographics()
    
    # logger.info("\n\n===== TESTING QUEUE MANAGEMENT ANALYSIS =====")
    # queue_results = test_queue_management() 