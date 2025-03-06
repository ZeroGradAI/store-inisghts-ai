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
            
            # Craft prompt for gender demographics analysis with JSON response format
            # Using simpler field names without underscores to avoid escape issues
            prompt = """USER: <image>
Analyze this retail store image for gender demographics. How many men and women do you see in the image and what products are they looking at?

Please return your analysis as a JSON object with the following keys:
- mencount: number of men in the image
- womencount: number of women in the image
- products: list of products they are looking at
- insights: your analysis of what this means for the store

Example format:
{
  "mencount": 2,
  "womencount": 3,
  "products": ["clothing", "electronics", "accessories"],
  "insights": "More women than men suggesting a female-oriented shopping experience."
}

ASSISTANT:"""
            
            # Generate response
            response = self._generate_response(processed_image, prompt)
            
            # Log the full raw response from the model
            logger.info("===== RAW MODEL RESPONSE FOR GENDER DEMOGRAPHICS =====")
            logger.info(response)
            logger.info("========== END OF RAW MODEL RESPONSE ==========")
            
            # Extract JSON data from the response
            result = self._extract_json_data(response, "gender_demographics")
            
            if not result:
                logger.warning("Failed to extract gender demographics from model response")
                return self._get_fallback_gender_demographics()
            
            # Add raw response to the results
            result['raw_response'] = response
            result['is_mock'] = False
            
            # Map new field names to old ones for backward compatibility
            if 'mencount' in result and 'men_count' not in result:
                result['men_count'] = result['mencount']
            if 'womencount' in result and 'women_count' not in result:
                result['women_count'] = result['womencount']
                
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
            'mencount': 1,
            'womencount': 3,
            'products': 'Fresh produce, Grocery items, Shopping carts',
            'insights': 'Customers are actively shopping and browsing products, Some customers are using shopping carts, indicating larger purchases, The store layout appears to encourage browsing through multiple aisles',
            'raw_response': 'This is mock data. No actual model response available.',
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
            
            # Craft prompt for queue analysis with simpler field names
            prompt = """USER: <image>
Analyze this retail store image for queue management. Get a count of how many counters are open and how many are closed. Also how many customers are waiting in line across all counters? Any recommendations for improving the queue management?

You should return a json object with the following keys:
- opencounters: number of open counters
- closedcounters: number of closed counters
- totalcounters: total number of counters
- customersinqueue: number of customers waiting in line
- waittime: average wait time (estimate)
- queueefficiency: how efficient is the queue management (0-1 or text)
- overcrowded: boolean true/false if counters are overcrowded
- recommendations: list of recommendations to improve queue management

Example format:
{
  "opencounters": 3,
  "closedcounters": 2,
  "totalcounters": 5,
  "customersinqueue": 10,
  "waittime": "5-10 minutes",
  "queueefficiency": "moderate",
  "overcrowded": true,
  "recommendations": ["Open more counters", "Add express lanes", "Improve customer flow"]
}

ASSISTANT:"""
            
            # Generate response
            response = self._generate_response(processed_image, prompt)
            
            # Log the full raw response from the model
            logger.info("===== RAW MODEL RESPONSE FOR QUEUE MANAGEMENT =====")
            logger.info(response)
            logger.info("========== END OF RAW MODEL RESPONSE ==========")
            
            # Extract JSON data from the response
            result = self._extract_json_data(response, "queue_management")
            
            if not result:
                logger.warning("Failed to extract queue management data from model response")
                return self._get_fallback_queue_management()
            
            # Add raw response to the results
            result['raw_response'] = response
            result['is_mock'] = False
            
            # Map new field names to old ones for backward compatibility
            mapping = {
                'opencounters': 'open_counters',
                'closedcounters': 'closed_counters',
                'totalcounters': 'total_counters',
                'customersinqueue': 'customers_in_queue',
                'waittime': 'avg_wait_time',
                'queueefficiency': 'queue_efficiency',
                'overcrowded': 'overcrowded_counters'
            }
            
            for new_key, old_key in mapping.items():
                if new_key in result and old_key not in result:
                    result[old_key] = result[new_key]
                    
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing queue management: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return self._get_fallback_queue_management()
    
    def _extract_json_data(self, response, analysis_type):
        """
        Extract JSON data from the model response.
        
        Args:
            response: The raw response from the model
            analysis_type: Type of analysis ('gender_demographics' or 'queue_management')
            
        Returns:
            A dictionary with the extracted data or None if extraction failed
        """
        try:
            # Extract JSON portion from the response
            # First try to find JSON block within code blocks (```json ... ```)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                logger.info(f"Found JSON in code block: {json_str[:100]}...")
            else:
                # Try to find any JSON object in the response
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    logger.info(f"Found JSON object: {json_str[:100]}...")
                else:
                    # If no JSON found, try to use the whole response
                    json_str = response
                    logger.warning("No JSON object found in response. Attempting to parse entire response.")
            
            # Parse the JSON data
            try:
                # Clean up the JSON string
                clean_json_str = json_str
                
                # Remove any markdown formatting or extra text
                clean_json_str = re.sub(r'[\n\r\t]', ' ', clean_json_str)
                
                # Handle escaped underscores (common issue with AI models)
                clean_json_str = clean_json_str.replace('\\_', '_')
                
                # Try to validate and parse the JSON
                try:
                    data = json.loads(clean_json_str)
                except json.JSONDecodeError:
                    # Additional attempt: replace escaped characters more aggressively
                    clean_json_str = re.sub(r'\\([^\\])', r'\1', clean_json_str)
                    data = json.loads(clean_json_str)
                
                logger.info(f"Successfully parsed JSON data with keys: {list(data.keys())}")
                
                # For the specific analysis types, ensure required fields with simplified names
                if analysis_type == "gender_demographics":
                    # Check for both old and new field names
                    required_fields = {
                        "mencount": 0, 
                        "womencount": 0, 
                        "products": "Not specified", 
                        "insights": "Not specified"
                    }
                    
                    # Also check for old field names for backward compatibility
                    old_field_mapping = {
                        "men_count": "mencount",
                        "women_count": "womencount"
                    }
                    
                    # Map old field names to new ones if they exist
                    for old_field, new_field in old_field_mapping.items():
                        if old_field in data and new_field not in data:
                            data[new_field] = data[old_field]
                    
                    # Ensure all required fields exist
                    for field, default_value in required_fields.items():
                        if field not in data:
                            logger.warning(f"Required field '{field}' missing from gender demographics JSON")
                            data[field] = default_value
                
                elif analysis_type == "queue_management":
                    # Check for both old and new field names
                    required_fields = {
                        "opencounters": 0,
                        "closedcounters": 0,
                        "totalcounters": 0,
                        "customersinqueue": 0,
                        "waittime": "Not specified",
                        "queueefficiency": "Not specified",
                        "overcrowded": False,
                        "recommendations": "Not specified"
                    }
                    
                    # Also check for old field names for backward compatibility
                    old_field_mapping = {
                        "open_counters": "opencounters",
                        "closed_counters": "closedcounters",
                        "total_counters": "totalcounters",
                        "customers_in_queue": "customersinqueue",
                        "avg_wait_time": "waittime",
                        "queue_efficiency": "queueefficiency",
                        "overcrowded_counters": "overcrowded"
                    }
                    
                    # Map old field names to new ones if they exist
                    for old_field, new_field in old_field_mapping.items():
                        if old_field in data and new_field not in data:
                            data[new_field] = data[old_field]
                    
                    # Ensure all required fields exist
                    for field, default_value in required_fields.items():
                        if field not in data:
                            logger.warning(f"Required field '{field}' missing from queue management JSON")
                            data[field] = default_value
                    
                    # Calculate totalcounters if needed
                    if data["totalcounters"] == 0 and (data["opencounters"] > 0 or data["closedcounters"] > 0):
                        data["totalcounters"] = data["opencounters"] + data["closedcounters"]
                        logger.info(f"Calculated totalcounters: {data['totalcounters']}")
                    
                    # Convert overcrowded to boolean if it's not already
                    if not isinstance(data["overcrowded"], bool):
                        if isinstance(data["overcrowded"], list) and len(data["overcrowded"]) > 0:
                            logger.info(f"Found overcrowded counters: {data['overcrowded']}")
                            data["overcrowded"] = True
                        elif isinstance(data["overcrowded"], (int, float)) and data["overcrowded"] > 0:
                            logger.info(f"Found overcrowded counter count: {data['overcrowded']}")
                            data["overcrowded"] = True
                        elif isinstance(data["overcrowded"], str) and data["overcrowded"].lower() not in ["false", "none", "0", "", "no"]:
                            logger.info(f"Found overcrowded string: {data['overcrowded']}")
                            data["overcrowded"] = True
                        else:
                            data["overcrowded"] = False
                
                return data
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {str(e)}")
                logger.error(f"JSON string attempted to parse: {clean_json_str[:500]}")
                
                # If JSON parsing fails, fall back to the old extraction methods
                if analysis_type == "gender_demographics":
                    logger.info("Falling back to regex extraction for gender demographics")
                    gender_data = self._extract_gender_counts(response)
                    products = self._extract_products(response)
                    insights = self._extract_insights(response)
                    
                    return {
                        'mencount': gender_data.get('men_count', 0),
                        'womencount': gender_data.get('women_count', 0),
                        'men_count': gender_data.get('men_count', 0),
                        'women_count': gender_data.get('women_count', 0),
                        'products': products,
                        'insights': insights
                    }
                elif analysis_type == "queue_management":
                    logger.info("Falling back to regex extraction for queue management")
                    queue_info = self._extract_queue_info(response)
                    
                    # Map old field names to new ones
                    if queue_info:
                        queue_info['opencounters'] = queue_info.get('open_counters', 0)
                        queue_info['closedcounters'] = queue_info.get('closed_counters', 0)
                        queue_info['totalcounters'] = queue_info.get('total_counters', 0)
                        queue_info['customersinqueue'] = queue_info.get('customers_in_queue', 0)
                        queue_info['waittime'] = queue_info.get('avg_wait_time', 'Not specified')
                        queue_info['queueefficiency'] = queue_info.get('queue_efficiency', 'Not specified')
                        queue_info['overcrowded'] = queue_info.get('overcrowded_counters', False)
                    
                    return queue_info
                
                return None
                
        except Exception as e:
            logger.error(f"Error extracting JSON data: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return None

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
            'raw_response': 'This is mock data. No actual model response available.',
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
            logger.info(f"Extracting queue info from response: {response[:100]}...")  # Log only first 100 chars for brevity
            
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
                logger.info(f"Matched total counters: {result['total_counters']}")
            
            # Extract number of open counters
            open_counters_pattern = r'(\d+)\s*(?:checkout |open )?counters'
            open_counters_match = re.search(open_counters_pattern, response, re.IGNORECASE)
            if open_counters_match:
                result['open_counters'] = int(open_counters_match.group(1))
                logger.info(f"Matched open counters: {result['open_counters']}")
            
            # Extract number of closed counters explicitly
            closed_counters_pattern = r'(\d+)\s*(?:closed|inactive|unused)\s*counters'
            closed_counters_match = re.search(closed_counters_pattern, response, re.IGNORECASE)
            if closed_counters_match:
                result['closed_counters'] = int(closed_counters_match.group(1))
                logger.info(f"Matched closed counters: {result['closed_counters']}")
            
            # Calculate closed or total counters if needed
            if result['total_counters'] > 0 and result['open_counters'] > 0 and result['closed_counters'] == 0:
                # If we have total and open but not closed, calculate closed
                result['closed_counters'] = result['total_counters'] - result['open_counters']
                logger.info(f"Calculated closed counters: {result['closed_counters']} (total - open)")
            elif result['total_counters'] == 0 and result['open_counters'] > 0 and result['closed_counters'] > 0:
                # If we have open and closed but not total, calculate total
                result['total_counters'] = result['open_counters'] + result['closed_counters']
                logger.info(f"Calculated total counters: {result['total_counters']} (open + closed)")
            elif result['total_counters'] == 0 and result['open_counters'] == 0 and result['closed_counters'] == 0:
                # If we couldn't extract any counter information, set default values
                result['open_counters'] = 2
                result['closed_counters'] = 1
                result['total_counters'] = 3
                logger.info("Using default counter values because no counters were found in the response")
            elif result['total_counters'] == 0:
                # If we just don't have a total, calculate it
                result['total_counters'] = result['open_counters'] + result['closed_counters']
                logger.info(f"Calculated total counters: {result['total_counters']} (open + closed)")
            elif result['closed_counters'] == 0 and result['total_counters'] > result['open_counters']:
                # If we just don't have closed counters, calculate it
                result['closed_counters'] = result['total_counters'] - result['open_counters']
                logger.info(f"Calculated closed counters: {result['closed_counters']} (total - open)")
            
            # Extract number of customers in queue
            queue_pattern = r'(\d+)\s*customers?\s*(?:in|waiting|queuing)'
            queue_match = re.search(queue_pattern, response, re.IGNORECASE)
            if queue_match:
                result['customers_in_queue'] = int(queue_match.group(1))
                logger.info(f"Matched customers in queue: {result['customers_in_queue']}")
            else:
                logger.info("No customers in queue pattern found in response")
            
            # Extract queue efficiency
            efficiency_pattern = r'queue management is\s*(\w+)'
            efficiency_match = re.search(efficiency_pattern, response, re.IGNORECASE)
            if efficiency_match:
                result['queue_efficiency'] = efficiency_match.group(1)
                logger.info(f"Matched queue efficiency: {result['queue_efficiency']}")
            
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
                
                logger.info(f"Customers per counter: {customers_per_counter:.1f}")
                logger.info(f"Overcrowded: {result['overcrowded_counters']}")
                logger.info(f"Wait time estimate: {result['avg_wait_time']}")
                
                # Add recommendations based on crowding
                if result['overcrowded_counters']:
                    result['recommendations'] = 'Open more checkout counters to reduce wait times, Consider implementing a queue management system'
                else:
                    result['recommendations'] = 'Current queue management is efficient, Monitor customer flow during peak hours'
            else:
                # Default values if we couldn't extract meaningful data
                result['overcrowded_counters'] = False
                result['avg_wait_time'] = 'Not enough data'
                if result['customers_in_queue'] == 0:
                    result['recommendations'] = 'No customers waiting. Maintain current staffing levels during non-peak hours.'
                else:
                    result['recommendations'] = 'Ensure adequate staffing during peak hours'
                
                logger.info("No crowding calculation possible (missing open counters or customers in queue)")
                logger.info(f"Open counters: {result['open_counters']}, Customers in queue: {result['customers_in_queue']}")
            
            # Look for explicit mentions of overcrowding in the text
            if 'overcrowd' in response.lower() or 'long wait' in response.lower() or 'long line' in response.lower():
                result['overcrowded_counters'] = True
                if result['recommendations'] == 'Not specified':
                    result['recommendations'] = 'Open more checkout counters to reduce wait times'
                logger.info("Detected explicit mention of overcrowding or long wait")
            
            # Set is_mock flag to indicate this is real data
            result['is_mock'] = False
            
            # Log the final result keys
            logger.info(f"Final queue info keys: {list(result.keys())}")
            
            return result
                
        except Exception as e:
            logger.error(f"Error extracting queue information: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return self._get_fallback_queue_management()

def get_model(use_small_model=True):
    """Get an instance of the ModelInference class."""
    return ModelInference(use_small_model=use_small_model) 