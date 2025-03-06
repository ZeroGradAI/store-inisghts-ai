"""
Self-contained minimal implementation of LLaVA functionality.
This avoids any imports from the actual LLaVA package.
"""

import torch
import os
import importlib
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import logging
import requests
from io import BytesIO
import sys

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("ModelInference")

# Hugging Face transformers imports
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor

# Constants from llava.constants
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

class SeparatorStyle:
    """Separator style for conversation templates."""
    SINGLE = 0
    TWO = 1
    MPT = 2
    PLAIN = 3
    LLAMA_2 = 4
    MISTRAL = 5
    CHATML = 6

class Conversation:
    """A class that manages conversation prompts and contexts."""
    def __init__(self, system="", roles=("USER", "ASSISTANT"), messages=None, offset=0, sep_style=SeparatorStyle.SINGLE, sep=",", sep2=None, version="v1"):
        self.system = system
        self.roles = roles
        self.messages = messages or []
        self.offset = offset
        self.sep_style = sep_style
        self.sep = sep
        self.sep2 = sep2 or sep
        self.version = version

    def get_prompt(self):
        """Get the prompt for generation."""
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if i == 0 and role == self.roles[0] and message.startswith("I need help"):
                    message = message.replace("I need help", "Help me")
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ": "
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            ret = "" if self.system == "" else self.system + " "
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if i == 0:
                        ret += self.roles[0] + ": " + message + self.sep2
                    else:
                        ret += self.roles[1] + ": " + message + self.sep
                else:
                    ret += self.roles[1] + ": "
            return ret
        elif self.sep_style == SeparatorStyle.MISTRAL:
            ret = "" if self.system == "" else self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = self.system 
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n" 
            return ret
        else:
            raise ValueError(f"Invalid separator style: {self.sep_style}")

    def append_message(self, role, message):
        """Append a message to the conversation."""
        self.messages.append([role, message])

    def copy(self):
        """Create a deep copy of the conversation."""
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

# Conversation templates
conv_templates = {
    "llava_v1": Conversation(
        system="",
        roles=("USER", "ASSISTANT"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="</s>",
    ),
    "llava_llama_2": Conversation(
        system="<s>[INST] <<SYS>>\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using your language and vision capabilities.\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.LLAMA_2,
        sep=" ",
        sep2=" [/INST] ",
    ),
    "mistral_instruct": Conversation(
        system="<s>",
        roles=("[INST] ", " [/INST]"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.MISTRAL,
        sep="</s>",
        sep2="",
    ),
    "chatml_direct": Conversation(
        system="<|im_start|>system\nYou are a helpful vision and language assistant.<|im_end|>\n",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>\n",
        sep2="<|im_end|>\n",
    ),
}

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def get_model_name_from_path(model_path):
    """Get the model name from the path."""
    if model_path.endswith("/"):
        model_path = model_path[:-1]
    return model_path.split("/")[-1]

def load_image(image_file):
    """Load an image from a file path or URL."""
    if image_file.startswith(('http://', 'https://')):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_images(image_files):
    """Load multiple images."""
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def process_images(images, image_processor, model_config):
    """Process a batch of images for model input."""
    images_tensor = []
    for image in images:
        # Apply CLIP processor
        processed = image_processor.preprocess(
            image, 
            return_tensors='pt'
        )['pixel_values'][0]
        images_tensor.append(processed)
    return torch.stack(images_tensor, dim=0)

def tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors=None):
    """Tokenize the text and replace the image token with the image token index."""
    prompt_chunks = prompt.split(DEFAULT_IMAGE_TOKEN)
    
    # Convert the prompt chunks into tokens
    tokenized_chunks = []
    for chunk in prompt_chunks:
        tokenized_chunk = tokenizer(chunk, return_tensors=return_tensors).input_ids
        tokenized_chunks.append(tokenized_chunk)
    
    # Combine the tokenized chunks with the image token index
    tokenized_list = []
    for i, chunk in enumerate(tokenized_chunks):
        tokenized_list.append(chunk)
        if i < len(tokenized_chunks) - 1:
            tokenized_list.append(torch.tensor([[image_token_index]]))
    
    # Concatenate all tensors
    input_ids = torch.cat(tokenized_list, dim=1)
    
    return input_ids[0] if return_tensors is None else input_ids

def load_pretrained_model(model_path, model_base=None, model_name=None, load_8bit=False, load_4bit=False):
    """Load the model from the given path."""
    # Disable torch init to avoid redundant initialization
    disable_torch_init()
    
    # Set environment variables to disable flash attention which is causing issues
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["TORCH_INIT_FLASH_ATTENTION"] = "0"  # Disable flash attention
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model configuration
    config = AutoConfig.from_pretrained(model_path)
    
    # Load the vision model (CLIP)
    if hasattr(config, "mm_vision_tower"):
        vision_tower = config.mm_vision_tower
        vision_model = CLIPVisionModel.from_pretrained(vision_tower)
        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
    else:
        vision_tower = "openai/clip-vit-large-patch14-336"
        vision_model = CLIPVisionModel.from_pretrained(vision_tower)
        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
    
    # Multiple approaches to loading the model
    try:
        print("Trying to load model with attn_implementation='eager'...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation='eager'  # Avoid using flash attention
        )
    except Exception as e:
        print(f"First attempt failed: {e}")
        try:
            print("Trying direct LlavaForConditionalGeneration loading...")
            from transformers import LlavaForConditionalGeneration
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation='eager'  # Avoid using flash attention
            )
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            try:
                print("Trying to load with low_cpu_mem_usage=True...")
                # Try with low CPU memory usage
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation='eager'
                )
            except Exception as e3:
                print(f"Third attempt failed: {e3}")
                # Final attempt with base LlamaForCausalLM
                try:
                    print("Attempting to use LlamaForCausalLM directly...")
                    from transformers import LlamaForCausalLM
                    model = LlamaForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                except Exception as e4:
                    print(f"All attempts failed. Last error: {e4}")
                    raise ValueError(f"Could not load model {model_path}. Please check model compatibility and requirements.")
    
    # Set the vision tower for the model if needed
    if hasattr(model, "vision_tower") and not model.vision_tower:
        model.vision_tower = vision_model
    
    # Configure model for evaluation
    model.eval()
    
    return tokenizer, model, image_processor, config.max_position_embeddings

def image_parser(args):
    """Parse image file paths"""
    return args.image_file.split(args.sep)

def eval_model(args):
    """Evaluate the model with the given prompt and image."""
    # Get model name from path
    model_name = get_model_name_from_path(args.model_path)
    
    # Parse image paths
    image_paths = [args.image_file]
    
    # Load images
    images = load_images(image_paths)
    
    # Load model
    try:
        logger.info(f"Loading model: {args.model_path}")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name
        )
        
        # Process images
        image_tensors = process_images(images, image_processor, model.config)
        
        # Prepare the prompt for the conversation
        if hasattr(args, 'conv_mode') and args.conv_mode:
            conv_mode = args.conv_mode
        elif model_name.startswith(("llava-v1.5", "llava-v1.6")):
            conv_mode = "llava_v1"
        elif model_name.startswith("llava-llama-2"):
            conv_mode = "llava_llama2"
        elif model_name.startswith("mistral"):
            conv_mode = "mistral"
        else:
            conv_mode = "llava_v0"
            
        # Set up conversation
        if conv_mode not in conv_templates:
            logger.warning(f"Conversation mode {conv_mode} not found, using default")
            conv_mode = "llava_v1"
            
        conv = conv_templates[conv_mode].copy()
        
        # Add default system prompt if it doesn't exist
        if not conv.system:
            conv.system = "You are a helpful vision and language assistant."
        
        # Prepare visual prompt
        if image_tensors is not None:
            if DEFAULT_IMAGE_TOKEN in args.query:
                query = args.query
            else:
                query = DEFAULT_IMAGE_TOKEN + '\n' + args.query
            
            conv.append_message(conv.roles[0], query)
        else:
            conv.append_message(conv.roles[0], args.query)
            
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Prepare for text generation
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        # Set stop_str based on conversation mode
        if conv_mode == "mistral":
            stop_str = conv.sep
        else:
            stop_str = conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2
            
        # Capture output in a list rather than printing
        stopping_criteria = StoppingCriteriaList()
        outputs = []
        
        # Text generation
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )
            
        # Process output
        output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        
        # If there's a stop string, truncate at that point
        if stop_str and stop_str in output:
            output = output[:output.find(stop_str)].strip()
            
        # Return the output
        return output
    
    except Exception as e:
        logger.error(f"Error in eval_model: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return f"Error: {str(e)}"

# Define a helper class for StoppingCriteriaList (needed by the model)
class StoppingCriteriaList:
    def __init__(self):
        self.criteria = []
    
    def __call__(self, *args, **kwargs):
        return any(criterion(*args, **kwargs) for criterion in self.criteria)
    
    def append(self, criterion):
        self.criteria.append(criterion) 