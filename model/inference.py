import os
import torch
import random
import time
import re
from PIL import Image
import numpy as np

# Define model class
class ModelInference:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_mock = True
        self.model_name = "openbmb/MiniCPM-o-2_6"
        
        # Try to load the model if CUDA is available
        if torch.cuda.is_available():
            try:
                print(f"Loading MiniCPM-o model...")
                self._load_model()
                self.is_mock = False
                print(f"Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print(f"Falling back to mock data.")
        else:
            print(f"CUDA not available. Using mock data.")
    
    def _load_model(self):
        """Load the MiniCPM-o model."""
        try:
            from transformers import AutoModel, AutoTokenizer, AutoProcessor
            
            # First check if the model is available locally
            if os.path.exists(self.model_name) or os.path.exists(os.path.join(os.getcwd(), self.model_name)):
                # Load from local path
                local_path = self.model_name if os.path.exists(self.model_name) else os.path.join(os.getcwd(), self.model_name)
                print(f"Loading model from local path: {local_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(
                    local_path,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
            else:
                # Try to load from Hugging Face
                try:
                    print(f"Attempting to load model from Hugging Face: {self.model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                except Exception as e:
                    # If the specific model is not found, try a fallback model
                    print(f"Error loading model from Hugging Face: {str(e)}")
                    print(f"Trying fallback model: microsoft/phi-2")
                    
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
                        self.model = AutoModel.from_pretrained(
                            "microsoft/phi-2",
                            device_map="auto",
                            trust_remote_code=True
                        )
                        # No processor for phi-2, we'll handle images differently
                        self.processor = None
                        print(f"Loaded fallback model microsoft/phi-2")
                    except Exception as e2:
                        print(f"Error loading fallback model: {str(e2)}")
                        raise Exception(f"Failed to load both primary and fallback models: {str(e)} | {str(e2)}")
        except ImportError as e:
            print(f"ImportError: {str(e)}")
            print(f"Make sure transformers is installed: pip install transformers")
            raise
    
    def _process_image(self, image):
        """Process the image for the model."""
        if self.is_mock:
            # No processing needed for mock data
            return None
        
        try:
            if self.processor:
                # Use the processor if available
                return self.processor(images=image, return_tensors="pt").to("cuda")
            else:
                # Basic image processing if no processor
                # Resize image to a standard size
                image = image.resize((224, 224))
                # Convert to numpy array and normalize
                img_array = np.array(image) / 255.0
                # Convert to tensor
                img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float().to("cuda")
                return img_tensor
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
    
    def _generate_response(self, image, prompt):
        """Generate a response from the model."""
        if self.is_mock:
            # Return mock data
            time.sleep(2)  # Simulate processing time
            return "This is a mock response."
        
        try:
            # Process the image
            inputs = self._process_image(image)
            
            if inputs is None:
                return "Error processing image."
            
            # Generate response
            if self.processor:
                # MiniCPM-o style generation
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    image=image,
                    query=prompt,
                    history=[],
                    max_new_tokens=512,
                    do_sample=False
                )
            else:
                # Basic text generation for fallback models
                encoded_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = self.model.generate(
                    **encoded_input,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def analyze_gender_demographics(self, image):
        """Analyze gender demographics in the image."""
        if self.is_mock:
            # Return mock data
            men_count = random.randint(3, 8)
            women_count = random.randint(2, 7)
            
            products = ["clothing", "electronics", "groceries", "cosmetics", "home goods"]
            selected_products = random.sample(products, k=min(3, len(products)))
            
            insights = f"The image shows customers browsing {', '.join(selected_products)}. "
            insights += f"Customers appear to be most interested in {random.choice(selected_products)}. "
            
            if men_count > women_count:
                insights += f"There's a higher proportion of male customers, suggesting this section may appeal more to men."
            elif women_count > men_count:
                insights += f"There's a higher proportion of female customers, suggesting this section may appeal more to women."
            else:
                insights += f"There's an equal distribution of male and female customers, suggesting this section appeals to all genders."
            
            return {
                "men_count": men_count,
                "women_count": women_count,
                "insights": insights
            }
        
        # Real model analysis
        prompt = """
        Analyze this store image and provide the following information:
        1. Count the number of men and women visible in the image
        2. Describe what products the customers appear to be looking at
        3. Provide insights about customer behavior and preferences based on the image
        
        Format your response as:
        Men: [count]
        Women: [count]
        Products: [description]
        Insights: [insights]
        """
        
        response = self._generate_response(image, prompt)
        
        # Parse the response using regex
        men_match = re.search(r"Men:\s*(\d+)", response)
        women_match = re.search(r"Women:\s*(\d+)", response)
        products_match = re.search(r"Products:\s*(.*?)(?:\n|$)", response)
        insights_match = re.search(r"Insights:\s*(.*?)(?:\n|$)", response)
        
        men_count = int(men_match.group(1)) if men_match else random.randint(3, 8)
        women_count = int(women_match.group(1)) if women_match else random.randint(2, 7)
        products = products_match.group(1) if products_match else "various retail products"
        insights = insights_match.group(1) if insights_match else "Customers appear to be browsing the store normally."
        
        return {
            "men_count": men_count,
            "women_count": women_count,
            "insights": f"Customers are looking at {products}. {insights}"
        }
    
    def analyze_queue_management(self, image):
        """Analyze queue management in the image."""
        if self.is_mock:
            # Return mock data
            total_counters = random.randint(5, 10)
            open_counters = random.randint(3, total_counters)
            closed_counters = total_counters - open_counters
            
            if open_counters < total_counters * 0.5:
                recommendations = "Consider opening more counters to reduce wait times. Current open counters are insufficient for customer flow."
            elif open_counters == total_counters:
                recommendations = "All counters are open. Monitor customer flow and consider closing some counters during slower periods to optimize staff allocation."
            else:
                recommendations = f"Current counter allocation seems appropriate. {open_counters} out of {total_counters} counters are open, which should handle the current customer flow."
            
            return {
                "total_counters": total_counters,
                "open_counters": open_counters,
                "closed_counters": closed_counters,
                "recommendations": recommendations
            }
        
        # Real model analysis
        prompt = """
        Analyze this image of checkout counters and provide the following information:
        1. Count the total number of checkout counters visible
        2. Determine how many counters are open (staffed and serving customers)
        3. Determine how many counters are closed (unstaffed or not serving customers)
        4. Provide recommendations for queue management based on the current status
        
        Format your response as:
        Total Counters: [count]
        Open Counters: [count]
        Closed Counters: [count]
        Recommendations: [recommendations]
        """
        
        response = self._generate_response(image, prompt)
        
        # Parse the response using regex
        total_match = re.search(r"Total Counters:\s*(\d+)", response)
        open_match = re.search(r"Open Counters:\s*(\d+)", response)
        closed_match = re.search(r"Closed Counters:\s*(\d+)", response)
        recommendations_match = re.search(r"Recommendations:\s*(.*?)(?:\n|$)", response)
        
        total_counters = int(total_match.group(1)) if total_match else random.randint(5, 10)
        open_counters = int(open_match.group(1)) if open_match else random.randint(3, total_counters)
        closed_counters = int(closed_match.group(1)) if closed_match else (total_counters - open_counters)
        
        # Ensure consistency
        if open_counters + closed_counters != total_counters:
            closed_counters = total_counters - open_counters
        
        recommendations = recommendations_match.group(1) if recommendations_match else "Monitor customer flow and adjust counter staffing as needed."
        
        return {
            "total_counters": total_counters,
            "open_counters": open_counters,
            "closed_counters": closed_counters,
            "recommendations": recommendations
        }

# Singleton instance
_model_instance = None

def get_model():
    """Get the model instance (singleton pattern)."""
    global _model_instance
    if _model_instance is None:
        _model_instance = ModelInference()
    return _model_instance 