"""
Minimal implementation of required LLaVA utilities for Store Insights AI.
This avoids import issues with the LLaVA repository.
"""

import os
import sys
import torch
import logging
import tempfile
import subprocess
from PIL import Image

logger = logging.getLogger(__name__)

def get_model_name_from_path(model_path):
    """Get the model name from the path."""
    if '/' in model_path:
        return model_path.split('/')[-1]
    return model_path

def eval_model_subprocess(model_path, query, image_file, temperature=0.2, top_p=0.7, max_new_tokens=512):
    """
    Evaluate the model using a subprocess call to avoid import issues.
    This executes the LLaVA run_llava.py script directly.
    """
    llava_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LLaVA")
    run_llava_path = os.path.join(llava_dir, "llava", "eval", "run_llava.py")
    
    if not os.path.exists(run_llava_path):
        raise FileNotFoundError(f"Could not find run_llava.py at {run_llava_path}")
    
    # Prepare the command
    cmd = [
        sys.executable,
        run_llava_path,
        "--model-path", model_path,
        "--image-file", image_file,
        "--query", query,
        "--temperature", str(temperature),
        "--max-new-tokens", str(max_new_tokens)
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run the command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Process returned non-zero exit code: {result.returncode}")
        logger.error(f"stderr: {result.stderr}")
        return f"Error running model: {result.stderr}"
    
    return result.stdout.strip()

def process_image_for_llava(image):
    """Process an image for use with LLaVA."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image")
    
    # Ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_image_path = temp_file.name
        image.save(temp_image_path)
        logger.info(f"Saved processed image to temporary file: {temp_image_path}")
    
    return temp_image_path 