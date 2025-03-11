import os
import json
import logging
import base64
from PIL import Image
from openai import OpenAI
import io
import re
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APIModelInference:
    """Class for model inference operations using vision models via DeepInfra API."""
    
    def __init__(self, model_type='phi'):
        """
        Initialize the model inference class.
        
        Args:
            model_type: Type of vision model to use ('phi', 'llama', or 'llama-90b')
        """
        self.is_mock = False
        self.model_type = model_type.lower()
        
        # Set the vision model based on the model_type
        if self.model_type == 'phi':
            self.vision_model_id = config.PHI_VISION_MODEL_ID
            logger.info(f"Using Microsoft Phi-4 model: {self.vision_model_id}")
        elif self.model_type == 'llama':
            self.vision_model_id = config.LLAMA_VISION_MODEL_ID
            logger.info(f"Using Llama 11B model: {self.vision_model_id}")
        elif self.model_type == 'llama-90b':
            self.vision_model_id = config.LLAMA_VISION_MODEL_ID_90B
            logger.info(f"Using Llama 90B model: {self.vision_model_id}")
        else:
            logger.warning(f"Unknown model type: {model_type}. Defaulting to Llama 11B model.")
            self.vision_model_id = config.LLAMA_VISION_MODEL_ID
            self.model_type = 'llama'
            
        self.text_model_id = config.TEXT_MODEL_ID
        self.max_tokens = config.MAX_TOKENS
        
        try:
            # Get API key from configuration
            try:
                api_key = config.get_api_key()
            except ValueError as e:
                logger.warning(f"{str(e)} Using fallback mechanism.")
                # Fallback to hardcoded key - not recommended for production
                api_key = "MS6VJVml9vT4jR7GuagGDlLz2YnKP3hw"
            
            # Initialize OpenAI client with DeepInfra endpoint
            self.client = OpenAI(
                api_key=api_key,
                base_url=config.DEEPINFRA_API_URL,
            )
            logger.info(f"DeepInfra API client initialized with vision model: {self.vision_model_id}")
            logger.info(f"DeepInfra API client initialized with text model: {self.text_model_id}")
        except Exception as e:
            logger.error(f"Error initializing DeepInfra API client: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            logger.warning("Falling back to mock data.")
            self.is_mock = True
    
    def _process_image(self, image_path=None, image=None):
        """Process an image for model consumption.
        
        Args:
            image_path: Path to the image file
            image: PIL Image object
            
        Returns a base64 encoded image string ready for API consumption.
        """
        if self.is_mock:
            return "mock_image_processed"
        
        try:
            if image is not None:
                if isinstance(image, Image.Image):
                    # Convert PIL Image to base64
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG")
                    return base64.b64encode(buffer.getvalue()).decode("utf-8")
                elif isinstance(image, str) and os.path.isfile(image):
                    # If a file path is provided as a string
                    with open(image, "rb") as image_file:
                        return base64.b64encode(image_file.read()).decode("utf-8")
                else:
                    logger.error(f"Invalid image format: {type(image)}")
                    return None
            elif image_path is not None:
                if os.path.isfile(image_path):
                    with open(image_path, "rb") as image_file:
                        return base64.b64encode(image_file.read()).decode("utf-8")
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
    
    def _generate_vision_response(self, image, prompt):
        """Generate a response based on the image and prompt using the Llama vision model API.
        
        Args:
            image: Image data (base64 encoded string or PIL Image)
            prompt: Text prompt for the model
            
        Returns:
            Model response as a string
        """
        if self.is_mock:
            return "This is a mock response for testing purposes."
        
        try:
            # Process the image if it's a PIL Image
            if isinstance(image, Image.Image):
                base64_image = self._process_image(image_path=None, image=image)
            elif isinstance(image, str) and os.path.isfile(image):
                base64_image = self._process_image(image_path=image)
            else:
                base64_image = image  # Assume it's already processed
            
            if base64_image is None:
                logger.error("Failed to process image")
                return "Error: Could not process the image."
            
            # Format the API request for vision model
            chat_completion = self.client.chat.completions.create(
                model=self.vision_model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            )
            
            # Extract the response
            response = chat_completion.choices[0].message.content
            logger.info(f"Vision model tokens: Prompt={chat_completion.usage.prompt_tokens}, Completion={chat_completion.usage.completion_tokens}")
            
            return response
        except Exception as e:
            logger.error(f"Error in generate_vision_response: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return f"Error generating vision response: {str(e)}"

    def _generate_text_response(self, prompt):
        """Generate a response based on text prompt using the Llama text model API.
        
        Args:
            prompt: Text prompt for the model
            
        Returns:
            Model response as a string
        """
        if self.is_mock:
            return "This is a mock response for testing purposes."
        
        try:
            # Format the API request for text model
            chat_completion = self.client.chat.completions.create(
                model=self.text_model_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=2000,
            )
            
            # Extract the response
            response = chat_completion.choices[0].message.content
            logger.info(f"Text model tokens: Prompt={chat_completion.usage.prompt_tokens}, Completion={chat_completion.usage.completion_tokens}")
            
            return response
        except Exception as e:
            logger.error(f"Error in generate_text_response: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return f"Error generating text response: {str(e)}"

    def analyze_gender_demographics(self, image):
        """
        Analyze the gender demographics in a retail store image.
        
        Args:
            image: The image to analyze (file path or PIL Image)
            
        Returns:
            A dictionary with gender demographics information.
        """
        logger.info("Starting gender demographics analysis with Llama model")
        
        if self.is_mock:
            logger.info("Using mock data for gender demographics analysis")
            return self._get_fallback_gender_demographics()
        
        try:
            # STEP 1: Use vision model to analyze the image
            vision_prompt = """

                            1. How many men and women customers do you see in the image. Respond something like "I see 10 men and 15 women customers in the image."
                            2. Identify what products or store sections these customers appear to be browsing or interested in. Respond something like "They appear to be interested in the electronics, clothing, and home goods sections."
                            3. Provide general insights about customer shopping patterns that might be useful for retail management

                            Please provide a short and concise response for each question.
                            """
            
            # Generate raw text analysis from vision model
            raw_response = self._generate_vision_response(image, vision_prompt)
            logger.info("Vision model raw response received")
            logger.info(f"Vision model raw response: {raw_response}")
            
            # STEP 2: Use text model to extract structured data from raw response
            text_prompt = f"""Based on the following retail business analysis, extract the gender demographics data into a JSON format.

Raw Analysis:
{raw_response}

Please return a JSON object with the following keys:
- mencount: estimated number of adult men customers in the image (use 0 if none or unclear)
- womencount: estimated number of adult women customers in the image (use 0 if none or unclear)
- products: list of products or sections customers appear to be interested in (empty array if none mentioned)
- insights: business insights about customer behavior in the retail environment (empty array if none provided)

Return ONLY the JSON object, nothing else. If the analysis doesn't provide certain information, use reasonable defaults (0 for counts, empty arrays for lists).
"""
            
            # Generate structured JSON response from text model
            json_response = self._generate_text_response(text_prompt)
            logger.info("Text model JSON response received")
            logger.info(f"Text model JSON response: {json_response}")
            # Extract data from the JSON response
            result = self._extract_json_data(json_response, "gender_demographics")
            
            if not result:
                logger.warning("Failed to parse JSON response, using raw response for extraction")
                result = self._extract_gender_counts_from_text(raw_response)
                
            # Add raw responses to result
            result['vision_raw_response'] = raw_response
            result['text_raw_response'] = json_response
            result['raw_response'] = raw_response  # Keep original key for backward compatibility
            result['is_mock'] = False
            
            return result
                
        except Exception as e:
            logger.error(f"Error analyzing gender demographics: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return self._get_fallback_gender_demographics()
    
    def analyze_queue_management(self, image):
        """
        Analyze the queue management in a retail store image.
        
        Args:
            image: The image to analyze (file path or PIL Image)
            
        Returns:
            A dictionary with queue management information.
        """
        logger.info("Starting queue management analysis with Llama model")
        
        if self.is_mock:
            logger.info("Using mock data for queue management analysis")
            return self._get_fallback_queue_management()
        
        try:
            # STEP 1: Use vision model to analyze the image
            vision_prompt = """
            1. How many open and closed counters do you see in the image? 
            2. How many customers are waiting in line in these counters?
            3. Suggest any recommendations to improve customer flow and reduce wait times

            Please provide a short and concise response for each question.
            """
            
            # Generate raw text analysis from vision model
            raw_response = self._generate_vision_response(image, vision_prompt)
            logger.info("Vision model raw response received")
            logger.info(f"Vision model raw response: {raw_response}")
            
            # STEP 2: Use text model to extract structured data from raw response
            text_prompt = f"""Based on the following retail business analysis focused on queue management, extract the data into a JSON format.

Raw Analysis:
{raw_response}

Please return a JSON object with the following keys:
- opencounters: number of open/staffed checkout counters (use 0 if none or unclear)
- closedcounters: number of closed/unstaffed checkout counters (use 0 if none or unclear)
- totalcounters: total number of checkout counters (sum of open and closed)
- customersinqueue: estimated number of customers waiting in line (use 0 if none or unclear)
- waittime: approximate wait time estimate based on the number of customers in line(e.g., "3-5 minutes", "10+ minutes")
- queueefficiency: assessment of overall queue management efficiency (text description or rating)
- overcrowded: boolean true/false indicating if the checkout area appears overcrowded
- recommendations: list of business suggestions to improve checkout efficiency

Return ONLY the JSON object, nothing else. If the analysis doesn't provide certain information, use reasonable defaults.
"""
            
            # Generate structured JSON response from text model
            json_response = self._generate_text_response(text_prompt)
            logger.info("Text model JSON response received")
            logger.info(f"Text model JSON response: {json_response}")
            # Extract data from the JSON response
            result = self._extract_json_data(json_response, "queue_management")
            
            if not result:
                logger.warning("Failed to parse JSON response, using raw response for extraction")
                result = self._extract_queue_info_from_text(raw_response)
            
            # Add raw responses to result
            result['vision_raw_response'] = raw_response
            result['text_raw_response'] = json_response
            result['raw_response'] = raw_response  # Keep original key for backward compatibility
            result['is_mock'] = False
            
            return result
                
        except Exception as e:
            logger.error(f"Error analyzing queue management: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return self._get_fallback_queue_management()
    
    def _extract_json_data(self, response, analysis_type):
        """
        Extract JSON data from the model response.
        
        Args:
            response: The model response text
            analysis_type: The type of analysis being performed
            
        Returns:
            Extracted data as a dictionary
        """
        try:
            logger.info(f"Extracting {analysis_type} data from response")
            # Log a snippet of the response for debugging
            response_preview = response[:200] + "..." if len(response) > 200 else response
            logger.info(f"Response preview: {response_preview}")
            
            # Try to find JSON in the response
            start_idx = response.find("{")
            end_idx = response.rfind("}")
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = response[start_idx:end_idx + 1]
                logger.info(f"Found JSON: {json_str[:200]}...")
                
                try:
                    data = json.loads(json_str)
                    logger.info(f"Successfully parsed JSON with keys: {list(data.keys())}")
                    
                    if analysis_type == "gender_demographics":
                        return self._extract_gender_counts(data)
                    elif analysis_type == "queue_management":
                        return self._extract_queue_info(data)
                    else:
                        return data
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {str(e)}")
                    logger.error(f"Invalid JSON string: {json_str}")
                    # Try a more lenient approach - look for smaller JSON objects
                    return self._extract_data_from_text(response, analysis_type)
            else:
                logger.warning(f"No JSON found in response for {analysis_type}")
                logger.info(f"Full response: {response}")
                return self._extract_data_from_text(response, analysis_type)
                
        except Exception as e:
            logger.error(f"Error extracting JSON data: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return None
            
    def _extract_data_from_text(self, text, analysis_type):
        """
        Extract data from text when JSON extraction fails.
        This is a fallback method to try to get structured data.
        """
        logger.info(f"Attempting to extract {analysis_type} data from plain text")
        
        if analysis_type == "gender_demographics":
            return self._extract_gender_counts_from_text(text)
        elif analysis_type == "queue_management":
            return self._extract_queue_info_from_text(text)
        else:
            logger.warning(f"No text extraction method for {analysis_type}")
            return None

    def _extract_gender_counts(self, data):
        """Extract gender counts from JSON data."""
        # Map from the keys in the prompt (mencount, womencount) to what the application expects (men_count, women_count)
        # Also look for alternative keys that the API might return
        
        # Handle products field that might be null, string, or array
        products = data.get('products', [])
        if products is None:
            products = []
        if isinstance(products, str):
            # If it's a string, convert to list
            products = [product.strip() for product in products.split(',') if product.strip()]
        
        # Handle insights field that might be null, string, or array
        insights = data.get('insights', "No specific insights available")
        if insights is None:
            insights = "No specific insights available"
        if isinstance(insights, list) and insights:
            # If it's a non-empty list, convert to string
            insights = '. '.join(insights)
        elif isinstance(insights, list) and not insights:
            # If it's an empty list
            insights = "No specific insights available"
            
        result = {
            'men_count': int(data.get('mencount', data.get('men_count', data.get('male_count', 0))) or 0),
            'women_count': int(data.get('womencount', data.get('women_count', data.get('female_count', 0))) or 0),
            'mencount': int(data.get('mencount', data.get('men_count', data.get('male_count', 0))) or 0),
            'womencount': int(data.get('womencount', data.get('women_count', data.get('female_count', 0))) or 0),
            'products': products,
            'insights': insights,
        }
        
        # Log the keys found in the data for debugging
        logger.info(f"Gender data keys available: {list(data.keys())}")
        logger.info(f"Extracted men_count: {result['men_count']}, women_count: {result['women_count']}")
        logger.info(f"Extracted products: {result['products']}")
        
        return result
    
    def _extract_gender_counts_from_text(self, text):
        """Extract gender counts from text response using regex patterns."""
        # Initialize with fallback values
        result = self._get_fallback_gender_demographics()
        result['raw_response'] = text
        
        # Only search for patterns if we have text to analyze
        if not text or "I don't feel safe" in text or "cannot assist" in text.lower():
            logger.warning("Safety filter triggered or empty response, using fallback values")
            return result
        
        try:
            # Look for mentions of men/males with numbers
            men_patterns = [
                r'(\d+)\s*(?:men|male|males|man)',  # "5 men", "10 males"
                r'(?:men|male|males|man)(?:\s*:)?\s*(\d+)',  # "men: 3", "males 4"
                r'number of men (?:is|:|about)?\s*(\d+)'  # "number of men is 3"
            ]
            
            for pattern in men_patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    # Take the first match that looks like a valid number
                    for match in matches:
                        try:
                            result['men_count'] = int(match)
                            result['mencount'] = int(match)
                            break
                        except ValueError:
                            continue
                    break
            
            # Look for mentions of women/females with numbers
            women_patterns = [
                r'(\d+)\s*(?:women|female|females|woman)',  # "5 women", "10 females"
                r'(?:women|female|females|woman)(?:\s*:)?\s*(\d+)',  # "women: 3", "females 4"
                r'number of women (?:is|:|about)?\s*(\d+)'  # "number of women is 3"
            ]
            
            for pattern in women_patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    # Take the first match that looks like a valid number
                    for match in matches:
                        try:
                            result['women_count'] = int(match)
                            result['womencount'] = int(match)
                            break
                        except ValueError:
                            continue
                    break
            
            # Extract products
            products_section = None
            if "product" in text.lower():
                # Find sections where products are mentioned
                product_patterns = [
                    r'products?(?:\s*:)?\s*([^\.]+)',  # "Products: X, Y, Z"
                    r'looking at ([^\.]+)',  # "looking at X, Y, Z"
                    r'interested in ([^\.]+)',  # "interested in X, Y, Z"
                    r'browsing ([^\.]+)'  # "browsing X, Y, Z"
                ]
                
                for pattern in product_patterns:
                    matches = re.findall(pattern, text.lower())
                    if matches:
                        products_section = matches[0]
                        break
                
                if products_section:
                    # Clean up and convert to list
                    products = [p.strip() for p in products_section.split(',')]
                    if products:
                        result['products'] = products
            
            # Extract insights
            if 'insight' in text.lower() or 'analysis' in text.lower() or 'suggest' in text.lower():
                # Look for complete sentences that might contain insights
                sentences = re.split(r'(?<=[.!?])\s+', text)
                insight_sentences = []
                
                for sentence in sentences:
                    if (len(sentence) > 20 and  # Reasonably complete thought
                        ('suggest' in sentence.lower() or 
                         'recommend' in sentence.lower() or
                         'indicate' in sentence.lower() or
                         'show' in sentence.lower() or
                         'mean' in sentence.lower())):
                        insight_sentences.append(sentence.strip())
                
                if insight_sentences:
                    result['insights'] = ' '.join(insight_sentences)
            
            logger.info(f"Extracted from text - men: {result['men_count']}, women: {result['women_count']}")
            
            # We have real data, so set is_mock to False
            result['is_mock'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting gender counts from text: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return result
    
    def _extract_queue_info(self, data):
        """Extract queue information from JSON data."""
        # Map from the keys in the prompt (opencounters, etc.) to what the application expects (open_counters, etc.)
        # Also look for alternative keys that the API might return
        
        # Handle recommendations field that might be null, string, or array
        recommendations = data.get('recommendations', [])
        if recommendations is None:
            recommendations = []
        if isinstance(recommendations, str):
            # If it's a string, convert to list
            recommendations = [rec.strip() for rec in recommendations.split(',') if rec.strip()]
        
        # Convert numeric fields to integers with safe fallbacks
        try:
            open_counters = int(data.get('opencounters', data.get('open_counters', 0)) or 0)
        except (ValueError, TypeError):
            open_counters = 0
            
        try:
            closed_counters = int(data.get('closedcounters', data.get('closed_counters', 0)) or 0)
        except (ValueError, TypeError):
            closed_counters = 0
            
        try:
            total_counters = int(data.get('totalcounters', data.get('total_counters', open_counters + closed_counters)) or 0)
        except (ValueError, TypeError):
            total_counters = open_counters + closed_counters
            
        try:
            customers_in_queue = int(data.get('customersinqueue', data.get('customers_in_queue', 0)) or 0)
        except (ValueError, TypeError):
            customers_in_queue = 0
        
        result = {
            'open_counters': open_counters,
            'closed_counters': closed_counters,
            'total_counters': total_counters,
            'customers_in_queue': customers_in_queue,
            'avg_wait_time': data.get('waittime', data.get('avg_wait_time', 'Unknown')),
            'queue_efficiency': data.get('queueefficiency', data.get('queue_efficiency', 'Unknown')),
            'overcrowded_counters': bool(data.get('overcrowded', data.get('overcrowded_counters', False))),
            'recommendations': recommendations
        }
        
        # Log the keys found in the data for debugging
        logger.info(f"Queue data keys available: {list(data.keys())}")
        logger.info(f"Extracted open_counters: {result['open_counters']}, closed_counters: {result['closed_counters']}")
        logger.info(f"Extracted customers_in_queue: {result['customers_in_queue']}")
        
        return result
    
    def _extract_queue_info_from_text(self, text):
        """Extract queue information from text response using regex patterns."""
        # Initialize with fallback values
        result = self._get_fallback_queue_management()
        result['raw_response'] = text
        
        # Only search for patterns if we have text to analyze
        if not text or "I don't feel safe" in text or "cannot assist" in text.lower():
            logger.warning("Safety filter triggered or empty response, using fallback values")
            return result
        
        try:
            # Look for mentions of open counters
            open_patterns = [
                r'(\d+)\s*(?:open|active|operational|staffed)(?:\s+counter|\s+checkout|\s+till|\s+cash register)',
                r'(?:open|active|operational|staffed)(?:\s+counter|\s+checkout|\s+till|\s+cash register)(?:s)?(?:\s*:)?\s*(\d+)',
                r'number of open counter(?:s)? (?:is|:|about)?\s*(\d+)'
            ]
            
            for pattern in open_patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    # Take the first match that looks like a valid number
                    for match in matches:
                        try:
                            if isinstance(match, tuple):
                                for m in match:
                                    if m and m.isdigit():
                                        result['open_counters'] = int(m)
                                        break
                            else:
                                result['open_counters'] = int(match)
                            break
                        except ValueError:
                            continue
                    break
            
            # Look for mentions of closed counters
            closed_patterns = [
                r'(\d+)\s*(?:closed|inactive|non-operational|unstaffed)(?:\s+counter|\s+checkout|\s+till|\s+cash register)',
                r'(?:closed|inactive|non-operational|unstaffed)(?:\s+counter|\s+checkout|\s+till|\s+cash register)(?:s)?(?:\s*:)?\s*(\d+)',
                r'number of closed counter(?:s)? (?:is|:|about)?\s*(\d+)'
            ]
            
            for pattern in closed_patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    # Take the first match that looks like a valid number
                    for match in matches:
                        try:
                            if isinstance(match, tuple):
                                for m in match:
                                    if m and m.isdigit():
                                        result['closed_counters'] = int(m)
                                        break
                            else:
                                result['closed_counters'] = int(match)
                            break
                        except ValueError:
                            continue
                    break
            
            # Calculate total counters
            total_counters_found = False
            total_patterns = [
                r'(\d+)\s*(?:total|in total|altogether)(?:\s+counter|\s+checkout|\s+till|\s+cash register)',
                r'(?:total|in total|altogether)(?:\s+counter|\s+checkout|\s+till|\s+cash register)(?:s)?(?:\s*:)?\s*(\d+)',
                r'total number of counter(?:s)? (?:is|:|about)?\s*(\d+)'
            ]
            
            for pattern in total_patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    # Take the first match that looks like a valid number
                    for match in matches:
                        try:
                            if isinstance(match, tuple):
                                for m in match:
                                    if m and m.isdigit():
                                        result['total_counters'] = int(m)
                                        total_counters_found = True
                                        break
                            else:
                                result['total_counters'] = int(match)
                                total_counters_found = True
                            break
                        except ValueError:
                            continue
                    break
            
            # If total wasn't explicitly mentioned, calculate from open + closed
            if not total_counters_found:
                result['total_counters'] = result['open_counters'] + result['closed_counters']
            
            # Look for mentions of customers in queue
            customer_patterns = [
                r'(\d+)\s*(?:customer|person|people|individual)(?:s)?(?:\s+in line|\s+waiting|\s+in queue)',
                r'(?:customer|person|people|individual)(?:s)?(?:\s+in line|\s+waiting|\s+in queue)(?:\s*:)?\s*(\d+)',
                r'number of (?:customer|person|people|individual)(?:s)? (?:in line|waiting|in queue) (?:is|:|about)?\s*(\d+)',
                r'queue length (?:is|:|about)?\s*(\d+)'
            ]
            
            for pattern in customer_patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    # Take the first match that looks like a valid number
                    for match in matches:
                        try:
                            if isinstance(match, tuple):
                                for m in match:
                                    if m and m.isdigit():
                                        result['customers_in_queue'] = int(m)
                                        break
                            else:
                                result['customers_in_queue'] = int(match)
                            break
                        except ValueError:
                            continue
                    break
            
            # Extract wait time estimates
            if 'wait time' in text.lower() or 'waiting time' in text.lower():
                wait_patterns = [
                    r'(?:wait|waiting) time (?:is|about|approximately|around|:)?\s*(\d+[-\s]?\d*\s*(?:minute|min|second|sec))',
                    r'(\d+[-\s]?\d*\s*(?:minute|min|second|sec)(?:s)?) (?:wait|waiting) time',
                    r'(?:wait|waiting) (?:is|about|approximately|around|:)?\s*(\d+[-\s]?\d*\s*(?:minute|min|second|sec))'
                ]
                
                for pattern in wait_patterns:
                    matches = re.findall(pattern, text.lower())
                    if matches:
                        result['avg_wait_time'] = matches[0].strip()
                        break
            
            # Extract queue efficiency assessment
            if 'efficien' in text.lower() or 'flow' in text.lower():
                efficiency_patterns = [
                    r'(?:queue|checkout|service) (?:is|seems|appears to be) (\w+)',
                    r'efficiency (?:is|:) (\w+)',
                    r'(\w+) efficiency'
                ]
                
                efficiency_keywords = {
                    'high': 'High', 'good': 'Good', 'moderate': 'Moderate', 'average': 'Average',
                    'low': 'Low', 'poor': 'Poor', 'excellent': 'Excellent', 'great': 'Great',
                    'efficient': 'Efficient', 'inefficient': 'Inefficient'
                }
                
                for pattern in efficiency_patterns:
                    matches = re.findall(pattern, text.lower())
                    if matches:
                        for match in matches:
                            for keyword, value in efficiency_keywords.items():
                                if keyword in match.lower():
                                    result['queue_efficiency'] = value
                                    break
                            if result['queue_efficiency'] != 'Unknown':
                                break
                        break
            
            # Check for overcrowding
            result['overcrowded_counters'] = (
                'overcrowd' in text.lower() or 
                'congest' in text.lower() or 
                'too many customer' in text.lower() or
                'long wait' in text.lower() or
                'long line' in text.lower()
            )
            
            # Extract recommendations
            if 'recommend' in text.lower() or 'suggest' in text.lower() or 'could' in text.lower():
                # Look for bullet points or numbered lists
                bullet_pattern = r'(?:(?:\d+[\.\)]\s*|\*\s*|\-\s*)[^\.]+[\.])'
                bullet_matches = re.findall(bullet_pattern, text)
                
                if bullet_matches:
                    result['recommendations'] = [m.strip() for m in bullet_matches]
                else:
                    # Try to extract sentences that sound like recommendations
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    rec_sentences = []
                    
                    for sentence in sentences:
                        if ('should' in sentence.lower() or 
                            'could' in sentence.lower() or 
                            'recommend' in sentence.lower() or 
                            'suggest' in sentence.lower() or
                            'consider' in sentence.lower() or
                            'implement' in sentence.lower() or
                            'improve' in sentence.lower()):
                            rec_sentences.append(sentence.strip())
                    
                    if rec_sentences:
                        result['recommendations'] = rec_sentences
            
            logger.info(f"Extracted from text - open: {result['open_counters']}, closed: {result['closed_counters']}, customers: {result['customers_in_queue']}")
            
            # We have real data, so set is_mock to False
            result['is_mock'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting queue info from text: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return result
    
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

    def get_model_name(self):
        """Return a human-readable name for the current model."""
        if self.model_type == 'phi':
            return "Microsoft Phi-4 Multimodal"
        elif self.model_type == 'llama':
            return "Llama-3.2-11B-Vision"
        elif self.model_type == 'llama-90b':
            return "Llama-3.2-90B-Vision"
        else:
            return "Unknown Model"

# Rename the get_model function to be more descriptive of what it's getting
def get_api_model(model_type=config.DEFAULT_MODEL):
    """
    Get an instance of the APIModelInference class.
    
    Args:
        model_type: Type of vision model to use ('phi' or 'llama')
        
    Returns:
        An instance of the APIModelInference class
    """
    return APIModelInference(model_type=model_type) 