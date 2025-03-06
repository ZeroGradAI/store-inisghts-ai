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
            'customers_in_queue': 4,
            'avg_wait_time': '3-5 minutes',
            'queue_efficiency': 'Moderate',
            'recommendations': 'Consider opening additional checkout lanes during peak hours, Implement express lanes for customers with fewer items',
            'is_mock': True
        }
    
    def _extract_gender_counts(self, response):
        """Extract gender counts from model response."""
        try:
            # Default values
            men_count = 0
            women_count = 0
            
            # Look for numbers of men and women in the response
            men_pattern = r'(\d+)\s*men'
            women_pattern = r'(\d+)\s*women'
            
            men_match = re.search(men_pattern, response, re.IGNORECASE)
            women_match = re.search(women_pattern, response, re.IGNORECASE)
            
            if men_match:
                men_count = int(men_match.group(1))
            
            if women_match:
                women_count = int(women_match.group(1))
            
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
            # Look for mentions of products
            products_pattern = r'(looking at|browsing|viewing|shopping for|examining|checking|exploring)\s+(.*?)(?:\.|$)'
            
            products_match = re.search(products_pattern, response, re.IGNORECASE)
            
            if products_match:
                return products_match.group(2).strip()
            else:
                # If no specific pattern match, use a broader approach
                words = response.split()
                for i, word in enumerate(words):
                    if word.lower() in ['products', 'product', 'items', 'item'] and i < len(words) - 1:
                        # Return the rest of the sentence after "products"
                        return ' '.join(words[i+1:]).split('.')[0]
            
            return "Not specified"
        except Exception as e:
            logger.error(f"Error extracting products: {str(e)}")
            return "Not specified"
    
    def _extract_insights(self, response):
        """Extract additional insights from model response."""
        # For now, return a basic insight based on the demographic split
        try:
            gender_data = self._extract_gender_counts(response)
            if gender_data:
                men = gender_data.get('men_count', 0)
                women = gender_data.get('women_count', 0)
                
                if men > women:
                    return f"More men ({men}) than women ({women}) in the store, suggesting male-oriented shopping preferences."
                elif women > men:
                    return f"More women ({women}) than men ({men}) in the store, suggesting female-oriented shopping preferences."
                else:
                    return f"Equal number of men and women ({men}) in the store, suggesting balanced shopping preferences."
            
            return "No specific insights available from this image."
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            return "No specific insights available from this image."
    
    def _extract_queue_info(self, response):
        """Extract queue management information from model response."""
        try:
            result = {
                'open_counters': 0,
                'customers_in_queue': 0,
                'avg_wait_time': 'Not specified',
                'queue_efficiency': 'Not specified',
                'recommendations': 'Not specified'
            }
            
            # Extract number of open counters
            counters_pattern = r'(\d+)\s*(?:checkout |open )?counters'
            counters_match = re.search(counters_pattern, response, re.IGNORECASE)
            if counters_match:
                result['open_counters'] = int(counters_match.group(1))
            
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
            
            # If we found at least some info, return the result
            if result['open_counters'] > 0 or result['customers_in_queue'] > 0:
                return result
            
            # If we couldn't extract structured information, include the full response
            result['full_response'] = response
            return result
            
        except Exception as e:
            logger.error(f"Error extracting queue information: {str(e)}")
            return None

def get_model(use_small_model=True):
    """Get an instance of the ModelInference class."""
    return ModelInference(use_small_model=use_small_model) 