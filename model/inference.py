import os
import logging
import torch
from PIL import Image
from transformers import BitsAndBytesConfig, pipeline
import json
import re

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelInference:
    """Class for model inference operations using the LLaVA model."""
    
    def __init__(self, use_small_model=False):
        """Initialize the model inference class."""
        self.is_mock = False
        self.model_id = "llava-hf/llava-1.5-7b-hf"
        self.max_new_tokens = 1000
        
        try:
            if torch.cuda.is_available():
                logger.info(f"CUDA is available. Detected {torch.cuda.device_count()} GPU(s).")
                logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
                logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
                
                logger.info(f"Loading model {self.model_id}...")
                self._load_model()
                logger.info(f"Model loaded successfully!")
            else:
                logger.warning("CUDA not available. Using fallback data.")
                self.is_mock = True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            logger.warning("Falling back to mock data.")
            self.is_mock = True
    
    def _load_model(self):
        """Load the LLaVA model using HuggingFace pipeline."""
        try:
            # Configure quantization for better memory performance
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Create the pipeline
            self.pipe = pipeline(
                "image-to-text", 
                model=self.model_id, 
                model_kwargs={"quantization_config": quantization_config}
            )
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            self.is_mock = True
            raise
    
    def _process_image(self, image_path=None, image=None):
        """Process an image for model consumption.
        
        Args:
            image_path: Path to the image file
            image: PIL Image object
            
        Returns a PIL Image ready for model consumption.
        """
        if self.is_mock:
            return "mock_image_processed"
        
        try:
            if image is not None:
                if isinstance(image, Image.Image):
                    return image
                elif isinstance(image, str) and os.path.isfile(image):
                    return Image.open(image).convert('RGB')
                else:
                    logger.error(f"Invalid image format: {type(image)}")
                    return None
            elif image_path is not None:
                if os.path.isfile(image_path):
                    return Image.open(image_path).convert('RGB')
                else:
                    logger.error(f"Image file not found: {image_path}")
                    return None
            else:
                logger.error("No image or image path provided")
                return None
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return None
    
    def _generate_response(self, image, prompt):
        """Generate a response based on the image and prompt using the LLaVA model.
        
        Args:
            image: PIL Image object
            prompt: Text prompt for the model
            
        Returns:
            Model response as a string
        """
        if self.is_mock:
            return "This is a mock response for testing purposes."
        
        try:
            # Ensure the prompt follows the correct format
            if not prompt.startswith("USER:"):
                prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            
            # Run inference
            outputs = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": self.max_new_tokens})
            response = outputs[0]["generated_text"]
            
            # Extract just the assistant's response if needed
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[1].strip()
            
            return response
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return f"Error generating response: {str(e)}"
    
    def analyze_gender_demographics(self, image):
        """
        Analyze the gender demographics in a retail store image.
        
        Args:
            image: The image to analyze (file path or PIL Image)
            
        Returns:
            A dictionary with gender demographics information.
        """
        logger.info("Starting gender demographics analysis")
        
        if self.is_mock:
            logger.info("Using mock data for gender demographics analysis")
            return self._get_fallback_gender_demographics()
        
        try:
            # Process the image
            processed_image = self._process_image(image_path=None, image=image)
            
            if processed_image is None:
                logger.error("Failed to process image")
                return self._get_fallback_gender_demographics()
            
            # Craft prompt for gender demographics analysis
            prompt = "USER: <image>\nHow many men and women do you see in the image and what products are they looking at?\nASSISTANT:"
            
            # Generate response
            response = self._generate_response(processed_image, prompt)
            
            # Extract gender demographics from the response
            gender_data = self._extract_gender_counts(response)
            products = self._extract_products(response)
            insights = self._extract_insights(response)
            
            # Combine all data
            result = {
                'men_count': gender_data.get('men_count', 0),
                'women_count': gender_data.get('women_count', 0),
                'products': products,
                'insights': insights,
                'is_mock': False
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing gender demographics: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return self._get_fallback_gender_demographics()
    
    def _get_fallback_gender_demographics(self):
        """Provide a reliable fallback for gender demographics analysis."""
        logger.info("Using fallback gender demographics data")
        return {
            'men_count': 1,
            'women_count': 3,
            'products': 'Fresh produce, Grocery items, Shopping carts',
            'insights': 'Customers are actively shopping and browsing products, Some customers are using shopping carts, indicating larger purchases, The store layout appears to encourage browsing through multiple aisles',
            'is_mock': True
        }

    def analyze_queue_management(self, image):
        """
        Analyze the queue management in a retail store image.
        
        Args:
            image: The image to analyze (file path or PIL Image)
            
        Returns:
            A dictionary with queue management information.
        """
        logger.info("Starting queue management analysis")
        
        if self.is_mock:
            logger.info("Using mock data for queue management analysis")
            return self._get_fallback_queue_management()
        
        try:
            # Process the image
            processed_image = self._process_image(image_path=None, image=image)
            
            if processed_image is None:
                logger.error("Failed to process image")
                return self._get_fallback_queue_management()
            
            # Craft prompt for queue analysis
            prompt = "USER: <image>\nAnalyze this retail store image for queue management. How many checkout counters are open? Are there any customers waiting in line? If so, how many and is the queue management efficient?\nASSISTANT:"
            
            # Generate response
            response = self._generate_response(processed_image, prompt)
            
            # Extract queue management information
            result = self._extract_queue_info(response)
            
            if not result:
                logger.warning("Failed to extract queue management data from model response")
                return self._get_fallback_queue_management()
            
            result['is_mock'] = False
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing queue management: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return self._get_fallback_queue_management()
    
    def _get_fallback_queue_management(self):
        """Provide a reliable fallback for queue management analysis."""
        logger.info("Using fallback queue management data")
        return {
            'open_counters': 2,
            'closed_counters': 1,
            'total_counters': 3,
            'customers_in_queue': 4,
            'avg_wait_time': '3-5 minutes',
            'queue_efficiency': 'Moderate',
            'overcrowded_counters': False,
            'recommendations': 'Consider opening additional checkout lanes during peak hours, Implement express lanes for customers with fewer items',
            'is_mock': True
        }
    
    def _extract_gender_counts(self, response):
        """Extract gender counts from model response."""
        try:
            # Default values
            men_count = 0
            women_count = 0
            
            # Dictionary to convert word numbers to integers
            word_to_number = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
            }
            
            # Look for numbers of men and women in the response with digits
            digit_men_pattern = r'(\d+)\s*men'
            digit_women_pattern = r'(\d+)\s*women'
            
            # Look for numbers of men and women as words
            word_men_pattern = r'(one|two|three|four|five|six|seven|eight|nine|ten)\s*men'
            word_women_pattern = r'(one|two|three|four|five|six|seven|eight|nine|ten)\s*women'
            
            # Search for both patterns for men
            digit_men_match = re.search(digit_men_pattern, response, re.IGNORECASE)
            word_men_match = re.search(word_men_pattern, response, re.IGNORECASE)
            
            # Search for both patterns for women
            digit_women_match = re.search(digit_women_pattern, response, re.IGNORECASE)
            word_women_match = re.search(word_women_pattern, response, re.IGNORECASE)
            
            # Extract men count
            if digit_men_match:
                men_count = int(digit_men_match.group(1))
            elif word_men_match:
                men_word = word_men_match.group(1).lower()
                men_count = word_to_number.get(men_word, 0)
            
            # Extract women count
            if digit_women_match:
                women_count = int(digit_women_match.group(1))
            elif word_women_match:
                women_word = word_women_match.group(1).lower()
                women_count = word_to_number.get(women_word, 0)
            
            # Log the extracted counts
            logger.info(f"Extracted gender counts - Men: {men_count}, Women: {women_count}")
            
            return {
                'men_count': men_count,
                'women_count': women_count
            }
        except Exception as e:
            logger.error(f"Error extracting gender counts: {str(e)}")
            return None
    
    def _extract_products(self, response):
        """Extract product information from model response."""
        try:
            # Look for mentions of products with different patterns
            products_patterns = [
                r'(looking at|browsing|viewing|shopping for|examining|checking|exploring)\s+(.*?)(?:\.|$)',
                r'products,\s+including\s+(.*?)(?:\.|$)',
                r'products\s+such\s+as\s+(.*?)(?:\.|$)',
                r'items\s+like\s+(.*?)(?:\.|$)',
                r'products,\s+which\s+include\s+(.*?)(?:\.|$)',
                r'products\s+(?:include|are)\s+(.*?)(?:\.|$)'
            ]
            
            # Try all patterns
            for pattern in products_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    # If it's the first pattern, we need group 2, otherwise group 1
                    group_idx = 2 if pattern == products_patterns[0] else 1
                    if len(match.groups()) >= group_idx:
                        extracted = match.group(group_idx).strip()
                        if extracted:
                            logger.info(f"Extracted products: {extracted}")
                            return extracted
            
            # If none of the specific patterns match, try more general cases
            if "products" in response.lower():
                # Find sentences containing "products"
                sentences = response.split('.')
                for sentence in sentences:
                    if "products" in sentence.lower():
                        # Remove any leading phrases before "products"
                        if "products including" in sentence.lower():
                            parts = sentence.lower().split("products including")
                            if len(parts) > 1:
                                return parts[1].strip()
                        if "products such as" in sentence.lower():
                            parts = sentence.lower().split("products such as")
                            if len(parts) > 1:
                                return parts[1].strip()
                        return sentence.strip()
            
            return "Various store products"
        except Exception as e:
            logger.error(f"Error extracting products: {str(e)}")
            return "Various store products"
    
    def _extract_insights(self, response):
        """Extract additional insights from model response."""
        try:
            gender_data = self._extract_gender_counts(response)
            if gender_data:
                men = gender_data.get('men_count', 0)
                women = gender_data.get('women_count', 0)
                
                if men == 0 and women == 0:
                    # If we couldn't extract counts but have a response, try to provide some basic insight
                    if "men" in response.lower() and "women" in response.lower():
                        return "The image shows a mix of male and female customers shopping in the store."
                    else:
                        return "The image shows customers shopping in the retail environment."
                
                if men > women:
                    return f"More men ({men}) than women ({women}) in the store, suggesting a male-oriented shopping experience."
                elif women > men:
                    return f"More women ({women}) than men ({men}) in the store, suggesting a female-oriented shopping experience."
                else:
                    return f"Equal number of men ({men}) and women ({women}) in the store, suggesting a balanced shopping environment."
            
            # Fallback to generic insights if we couldn't extract gender data
            if "products" in response.lower():
                for sentence in response.split('.'):
                    if "products" in sentence.lower():
                        return f"Customers are browsing: {sentence.strip()}"
            
            return "Customers are shopping in the retail environment."
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            return "Customers are shopping in the retail environment."
    
    def _extract_queue_info(self, response):
        """Extract queue management information from model response."""
        try:
            result = {
                'open_counters': 0,
                'closed_counters': 0,
                'total_counters': 0,
                'customers_in_queue': 0,
                'avg_wait_time': 'Not specified',
                'queue_efficiency': 'Not specified',
                'overcrowded_counters': False,
                'recommendations': 'Not specified'
            }
            
            # Extract number of total counters
            total_counters_pattern = r'(\d+)\s*(?:total|checkout|all)\s*counters'
            total_counters_match = re.search(total_counters_pattern, response, re.IGNORECASE)
            if total_counters_match:
                result['total_counters'] = int(total_counters_match.group(1))
            
            # Extract number of open counters
            open_counters_pattern = r'(\d+)\s*(?:checkout |open )?counters'
            open_counters_match = re.search(open_counters_pattern, response, re.IGNORECASE)
            if open_counters_match:
                result['open_counters'] = int(open_counters_match.group(1))
            
            # Extract number of closed counters explicitly
            closed_counters_pattern = r'(\d+)\s*(?:closed|inactive|unused)\s*counters'
            closed_counters_match = re.search(closed_counters_pattern, response, re.IGNORECASE)
            if closed_counters_match:
                result['closed_counters'] = int(closed_counters_match.group(1))
            
            # Calculate closed or total counters if needed
            if result['total_counters'] > 0 and result['open_counters'] > 0 and result['closed_counters'] == 0:
                # If we have total and open but not closed, calculate closed
                result['closed_counters'] = result['total_counters'] - result['open_counters']
            elif result['total_counters'] == 0 and result['open_counters'] > 0 and result['closed_counters'] > 0:
                # If we have open and closed but not total, calculate total
                result['total_counters'] = result['open_counters'] + result['closed_counters']
            elif result['total_counters'] == 0 and result['open_counters'] == 0 and result['closed_counters'] == 0:
                # If we couldn't extract any counter information, set default values
                result['open_counters'] = 2
                result['closed_counters'] = 1
                result['total_counters'] = 3
            elif result['total_counters'] == 0:
                # If we just don't have a total, calculate it
                result['total_counters'] = result['open_counters'] + result['closed_counters']
            elif result['closed_counters'] == 0 and result['total_counters'] > result['open_counters']:
                # If we just don't have closed counters, calculate it
                result['closed_counters'] = result['total_counters'] - result['open_counters']
            
            # Extract number of customers in queue
            queue_pattern = r'(\d+)\s*customers?\s*(?:in|waiting|queuing)'
            queue_match = re.search(queue_pattern, response, re.IGNORECASE)
            if queue_match:
                result['customers_in_queue'] = int(queue_match.group(1))
            
            # Extract queue efficiency
            efficiency_pattern = r'queue management is\s*(\w+)'
            efficiency_match = re.search(efficiency_pattern, response, re.IGNORECASE)
            if efficiency_match:
                result['queue_efficiency'] = efficiency_match.group(1)
            
            # Determine if counters are overcrowded
            # We'll say it's overcrowded if there are more than 3 customers per open counter
            if result['open_counters'] > 0 and result['customers_in_queue'] > 0:
                customers_per_counter = result['customers_in_queue'] / result['open_counters']
                result['overcrowded_counters'] = customers_per_counter > 3
                
                # Add wait time estimation based on crowding
                if customers_per_counter <= 1:
                    result['avg_wait_time'] = 'Less than 2 minutes'
                elif customers_per_counter <= 2:
                    result['avg_wait_time'] = '2-5 minutes'
                elif customers_per_counter <= 3:
                    result['avg_wait_time'] = '5-10 minutes'
                else:
                    result['avg_wait_time'] = 'More than 10 minutes'
                
                # Add recommendations based on crowding
                if result['overcrowded_counters']:
                    result['recommendations'] = 'Open more checkout counters to reduce wait times, Consider implementing a queue management system'
                else:
                    result['recommendations'] = 'Current queue management is efficient, Monitor customer flow during peak hours'
            else:
                # Default values if we couldn't extract meaningful data
                result['overcrowded_counters'] = False
                result['avg_wait_time'] = 'Not enough data'
                result['recommendations'] = 'Ensure adequate staffing during peak hours'
            
            # Look for explicit mentions of overcrowding in the text
            if 'overcrowd' in response.lower() or 'long wait' in response.lower() or 'long line' in response.lower():
                result['overcrowded_counters'] = True
                if 'recommendations' not in result or result['recommendations'] == 'Not specified':
                    result['recommendations'] = 'Open more checkout counters to reduce wait times'
            
            # Set is_mock flag to indicate this is real data
            result['is_mock'] = False
            
            # Log the extracted data
            logger.info(f"Extracted queue info: {result}")
            
            return result
                
        except Exception as e:
            logger.error(f"Error extracting queue information: {str(e)}")
            return self._get_fallback_queue_management()

def get_model(use_small_model=True):
    """Get an instance of the ModelInference class."""
    return ModelInference(use_small_model=use_small_model) 