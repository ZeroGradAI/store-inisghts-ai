"""
Self-contained minimal implementation of LLaVA functionality.
This avoids any imports from the actual LLaVA package.
"""

import os
import re
import torch
import logging
import requests
from io import BytesIO
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, CLIPVisionModel, CLIPImageProcessor

# Configure logging
logger = logging.getLogger(__name__)

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
    
    # Load the main model
    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", load_in_8bit=True
        )
    elif load_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", load_in_4bit=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
    
    # Set the vision tower for the model
    if hasattr(model, "vision_tower") and not model.vision_tower:
        model.vision_tower = vision_model
    
    # Configure model for evaluation
    model.eval()
    
    return tokenizer, model, image_processor, config.max_position_embeddings

def image_parser(args):
    """Parse image file paths"""
    return args.image_file.split(args.sep)

def eval_model(args):
    """Evaluate the model on the given image and prompt."""
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs) 