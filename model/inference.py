import os
import torch
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Union, Tuple
import re

# Import transformers only if CUDA is available to avoid errors on CPU-only machines
if torch.cuda.is_available():
    from transformers import AutoModel, AutoTokenizer

# This file will contain the actual model inference code when deployed
# For now, it provides the structure but will be implemented when deployed to a GPU environment

class MiniCPMoModel:
    """
    Interface for the MiniCPM-o model integration.
    Uses the actual model when CUDA is available, otherwise falls back to mock data.
    """
    
    def __init__(self, model_path: str = "openbmb/MiniCPM-o-2_6"):
        """
        Initialize the MiniCPM-o model.
        
        Args:
            model_path: Path to the model weights or HuggingFace model name
        """
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Load the actual model if CUDA is available
        if torch.cuda.is_available():
            try:
                print(f"Loading MiniCPM-o model from {model_path}...")
                self.model = AutoModel.from_pretrained(
                    model_path, 
                    trust_remote_code=True, 
                    torch_dtype=torch.bfloat16,
                    init_audio=False,
                    init_tts=False
                )
                self.model = self.model.to(device=self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model.eval()
                print(f"Model loaded successfully on {self.device}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            print(f"CUDA not available. Using mock data for inference.")

    def analyze_gender_demographics(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze an image to detect people and their gender.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not torch.cuda.is_available() or self.model is None:
            # For development without GPU, return mock results
            return self._mock_gender_analysis(image)
        
        # Prepare the image for the model
        processed_image = self._preprocess_image(image)
        
        # Create the prompt for gender demographics analysis
        prompt = "Analyze this image and tell me how many men and women are in it. Also describe what products they appear to be looking at or interested in."
        
        # Run model inference
        msgs = [
            {"role": "user", "content": [processed_image, prompt]}
        ]
        
        params = {
            'sampling': True,
            'top_p': 0.8,
            'top_k': 100,
            'temperature': 0.7,
            'repetition_penalty': 1.05,
            "max_new_tokens": 1024
        }
        
        try:
            answer = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                **params
            )
            
            # Clean up the response
            answer = re.sub(r'(<box>.*</box>)', '', answer)
            answer = answer.replace('<ref>', '').replace('</ref>', '')
            answer = answer.replace('<box>', '').replace('</box>', '')
            
            # Extract counts and description from the answer
            men_count = 0
            women_count = 0
            
            # Try to extract counts from the text
            men_match = re.search(r'(\d+)\s+men', answer, re.IGNORECASE)
            women_match = re.search(r'(\d+)\s+women', answer, re.IGNORECASE)
            
            if men_match:
                men_count = int(men_match.group(1))
            if women_match:
                women_count = int(women_match.group(1))
            
            return {
                'men_count': men_count,
                'women_count': women_count,
                'description': answer,
                'image': image
            }
        except Exception as e:
            print(f"Error during model inference: {e}")
            return self._mock_gender_analysis(image)
    
    def analyze_queue_management(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze an image to detect checkout counters and their status.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not torch.cuda.is_available() or self.model is None:
            # For development without GPU, return mock results
            return self._mock_queue_analysis(image)
        
        # Prepare the image for the model
        processed_image = self._preprocess_image(image)
        
        # Create the prompt for queue management analysis
        prompt = "Analyze this image of checkout counters in a store. Count the total number of counters, how many are open and closed, and identify any overcrowded counters. Provide suggestions for improving customer flow."
        
        # Run model inference
        msgs = [
            {"role": "user", "content": [processed_image, prompt]}
        ]
        
        params = {
            'sampling': True,
            'top_p': 0.8,
            'top_k': 100,
            'temperature': 0.7,
            'repetition_penalty': 1.05,
            "max_new_tokens": 1024
        }
        
        try:
            answer = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                **params
            )
            
            # Clean up the response
            answer = re.sub(r'(<box>.*</box>)', '', answer)
            answer = answer.replace('<ref>', '').replace('</ref>', '')
            answer = answer.replace('<box>', '').replace('</box>', '')
            
            # Extract counts and information from the answer
            total_counters = 0
            open_counters = 0
            closed_counters = 0
            overcrowded = []
            suggestions = answer
            
            # Try to extract counts from the text
            total_match = re.search(r'(\d+)\s+(?:total|checkout)\s+counters', answer, re.IGNORECASE)
            open_match = re.search(r'(\d+)\s+(?:open|active)\s+counters', answer, re.IGNORECASE)
            closed_match = re.search(r'(\d+)\s+(?:closed|inactive)\s+counters', answer, re.IGNORECASE)
            
            if total_match:
                total_counters = int(total_match.group(1))
            if open_match:
                open_counters = int(open_match.group(1))
            if closed_match:
                closed_counters = int(closed_match.group(1))
            
            # If we have open and closed but no total, calculate it
            if total_counters == 0 and open_counters > 0 and closed_counters > 0:
                total_counters = open_counters + closed_counters
            
            # Try to extract overcrowded counters
            overcrowded_match = re.search(r'(?:overcrowded|crowded).*?(\d+(?:,\s*\d+)*)', answer, re.IGNORECASE)
            if overcrowded_match:
                overcrowded_str = overcrowded_match.group(1)
                overcrowded = [int(num.strip()) for num in overcrowded_str.split(',')]
            
            # Extract suggestions
            suggestion_match = re.search(r'(?:suggest|recommend|advice).*?([^.]*\.)', answer, re.IGNORECASE)
            if suggestion_match:
                suggestions = suggestion_match.group(0)
            
            return {
                'total_counters': total_counters,
                'open_counters': open_counters,
                'closed_counters': closed_counters,
                'overcrowded': overcrowded,
                'suggestions': suggestions,
                'image': image
            }
        except Exception as e:
            print(f"Error during model inference: {e}")
            return self._mock_queue_analysis(image)
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess an image for the model.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Processed image ready for the model
        """
        # Resize image if needed (following the reference code)
        max_size = 448 * 16  # Same as in the reference code
        if max(image.size) > max_size:
            w, h = image.size
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        
        return image
    
    def _mock_gender_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """
        Generate mock gender analysis results for development.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary with mock analysis results
        """
        men_count = np.random.randint(2, 8)
        women_count = np.random.randint(2, 8)
        
        products = ["groceries", "clothing", "electronics", "home goods", "personal care items"]
        selected_products = np.random.choice(products, size=np.random.randint(1, 3), replace=False)
        
        description = f"The image shows {men_count + women_count} customers browsing the store. "
        description += f"There are {men_count} men and {women_count} women. "
        description += f"Customers appear to be looking at {', '.join(selected_products)}."
        
        return {
            'men_count': men_count,
            'women_count': women_count,
            'description': description,
            'image': image
        }
    
    def _mock_queue_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """
        Generate mock queue analysis results for development.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary with mock analysis results
        """
        total_counters = np.random.randint(5, 10)
        open_counters = np.random.randint(3, total_counters)
        closed_counters = total_counters - open_counters
        
        # Generate random overcrowded counters
        overcrowded = []
        if open_counters > 0:
            num_overcrowded = np.random.randint(0, min(3, open_counters))
            if num_overcrowded > 0:
                overcrowded = np.random.choice(range(1, open_counters + 1), size=num_overcrowded, replace=False).tolist()
        
        # Generate suggestion based on the analysis
        if len(overcrowded) > 0 and closed_counters > 0:
            suggestions = f"There {'are' if len(overcrowded) > 1 else 'is'} {len(overcrowded)} overcrowded {'counters' if len(overcrowded) > 1 else 'counter'}. Consider opening {min(len(overcrowded), closed_counters)} additional {'counters' if min(len(overcrowded), closed_counters) > 1 else 'counter'} to reduce wait times."
        elif len(overcrowded) > 0 and closed_counters == 0:
            suggestions = f"There {'are' if len(overcrowded) > 1 else 'is'} {len(overcrowded)} overcrowded {'counters' if len(overcrowded) > 1 else 'counter'}, but all counters are already open. Consider adding more staff to existing counters or expanding checkout capacity."
        else:
            suggestions = f"All {open_counters} open counters are operating efficiently. No changes needed at this time."
        
        return {
            'total_counters': total_counters,
            'open_counters': open_counters,
            'closed_counters': closed_counters,
            'overcrowded': overcrowded,
            'suggestions': suggestions,
            'image': image
        }

# Singleton instance for use throughout the application
model = MiniCPMoModel()

def get_model() -> MiniCPMoModel:
    """
    Get the singleton model instance.
    
    Returns:
        MiniCPMo model instance
    """
    return model 