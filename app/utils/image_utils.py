import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Union, List

def resize_image(image: Image.Image, max_size: int = 800) -> Image.Image:
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image: PIL Image to resize
        max_size: Maximum size for the largest dimension
        
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    
    # Calculate the new dimensions
    if width > height:
        if width > max_size:
            new_width = max_size
            new_height = int(height * (max_size / width))
    else:
        if height > max_size:
            new_height = max_size
            new_width = int(width * (max_size / height))
    
    # Resize the image
    return image.resize((new_width, new_height), Image.LANCZOS)

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to an OpenCV image (numpy array).
    
    Args:
        pil_image: PIL Image
        
    Returns:
        OpenCV image (numpy array)
    """
    # Convert PIL Image to numpy array
    numpy_image = np.array(pil_image)
    
    # Convert RGB to BGR (OpenCV format)
    if len(numpy_image.shape) == 3 and numpy_image.shape[2] == 3:
        return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
    return numpy_image

def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert an OpenCV image (numpy array) to a PIL Image.
    
    Args:
        cv2_image: OpenCV image (numpy array)
        
    Returns:
        PIL Image
    """
    # Convert BGR to RGB (PIL format)
    if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array to PIL Image
    return Image.fromarray(cv2_image)

def save_temp_image(image: Image.Image, prefix: str = "temp") -> str:
    """
    Save an image to a temporary file and return the path.
    
    Args:
        image: PIL Image to save
        prefix: Prefix for the temporary file name
        
    Returns:
        Path to the saved image
    """
    # Create a 'temp' directory if it doesn't exist
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate a unique filename
    filename = f"{prefix}_{np.random.randint(10000)}.jpg"
    file_path = os.path.join(temp_dir, filename)
    
    # Save the image
    image.save(file_path, "JPEG")
    
    return file_path

def delete_temp_image(file_path: str) -> None:
    """
    Delete a temporary image file.
    
    Args:
        file_path: Path to the image file to delete
    """
    if os.path.exists(file_path):
        os.remove(file_path)

def get_sample_images_paths(category: str = None) -> List[str]:
    """
    Get paths to sample images for testing.
    
    Args:
        category: Category of images to get (gender or queue)
        
    Returns:
        List of paths to sample images
    """
    # Base directory for sample images
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "samples")
    
    # Specific directory based on category
    if category == "gender":
        sample_dir = os.path.join(base_dir, "gender")
    elif category == "queue":
        sample_dir = os.path.join(base_dir, "queue")
    else:
        sample_dir = base_dir
    
    # Create the directory if it doesn't exist
    os.makedirs(sample_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = [".jpg", ".jpeg", ".png"]
    image_paths = []
    
    if os.path.exists(sample_dir):
        for file in os.listdir(sample_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(sample_dir, file))
    
    return image_paths 