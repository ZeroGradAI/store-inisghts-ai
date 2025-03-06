import sys
import os
import argparse
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, pipeline

# For testing the refactored module
try:
    from model.inference import ModelInference
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

def test_direct_pipeline():
    """Test the direct pipeline implementation."""
    print("Testing direct pipeline implementation...")
    
    # Ensure the image exists
    image_path = "store_image.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist.")
        print("Please provide a valid image path using --image argument.")
        return False
    
    image = Image.open(image_path).convert('RGB')

    # Configure model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model_id = "llava-hf/llava-1.5-7b-hf"
    max_new_tokens = 1000
    prompt = "USER: <image>\nHow many men and women do you see in the image and what products are they looking at??\nASSISTANT:"

    try:
        print(f"Loading model {model_id}...")
        pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})
        print("Model loaded successfully!")
        
        print("Running inference...")
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})
        
        print("\nRESULTS:")
        print(f"USER:\n{prompt.split('USER: ')[1].split('ASSISTANT:')[0].strip()}")
        print(f"ASSISTANT:\n{outputs[0]['generated_text']}")
        return True
    except Exception as e:
        print(f"Error in direct pipeline test: {str(e)}")
        return False

def test_refactored_module(image_path):
    """Test the refactored ModelInference module."""
    if not MODEL_AVAILABLE:
        print("ModelInference module not available to test.")
        return False
    
    print("\nTesting refactored ModelInference module...")
    
    try:
        # Initialize model
        print("Initializing ModelInference...")
        model = ModelInference(use_small_model=False)
        
        # Process image and generate response
        print("Analyzing gender demographics...")
        results = model.analyze_gender_demographics(image_path)
        
        print("\nRESULTS:")
        print(f"Men count: {results.get('men_count', 'Not found')}")
        print(f"Women count: {results.get('women_count', 'Not found')}")
        print(f"Products: {results.get('products', 'Not found')}")
        print(f"Insights: {results.get('insights', 'Not found')}")
        print(f"Is mock: {results.get('is_mock', 'Not found')}")
        
        return True
    except Exception as e:
        print(f"Error in refactored module test: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test LLaVA model implementations')
    parser.add_argument('--mode', type=str, default='both', choices=['direct', 'module', 'both'],
                        help='Which implementation to test (direct, module, or both)')
    parser.add_argument('--image', type=str, default='store_image.jpg',
                        help='Path to the image file to use for testing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Warning: Image file {args.image} does not exist.")
        while True:
            response = input("Do you want to continue with the default image path? (y/n): ")
            if response.lower() == 'y':
                break
            elif response.lower() == 'n':
                new_path = input("Enter the path to your image file: ")
                if os.path.exists(new_path):
                    args.image = new_path
                    break
                else:
                    print(f"Error: Image file {new_path} does not exist.")
            else:
                print("Please enter 'y' or 'n'.")
    
    # Determine which tests to run
    run_direct = args.mode in ['direct', 'both']
    run_module = args.mode in ['module', 'both']
    
    results = []
    
    # Run the tests
    if run_direct:
        direct_result = test_direct_pipeline()
        results.append(("Direct Pipeline", direct_result))
    
    if run_module:
        module_result = test_refactored_module(args.image)
        results.append(("Refactored Module", module_result))
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY:")
    print("="*50)
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{name}: {status}")
    print("="*50)