# Feature Design: Store Insights AI with LLaVA

## Overview

Store Insights AI is a Streamlit application designed to help retail store managers gain actionable insights from store camera images. The application uses the LLaVA (Large Language and Vision Assistant) model to analyze images and extract valuable information about customer demographics and queue management.

## Architecture

### Components

1. **Frontend (Streamlit)**
   - Dashboard: Displays aggregated insights and metrics
   - Gender Demographics Module: Analyzes customer gender distribution
   - Queue Management Module: Analyzes checkout counters and queues

2. **Model Interface (inference.py)**
   - Handles loading and inference with the LLaVA model
   - Processes images for model consumption
   - Parses model responses to extract structured data
   - Provides fallback mechanisms for when model inference fails

3. **LLaVA Model**
   - Vision-language model with 7B parameters
   - Capable of understanding and responding to queries about images
   - Runs on CUDA-enabled GPU for efficient inference

### Data Flow

1. User uploads an image through the Streamlit interface
2. Image is passed to the appropriate analysis function
3. The ModelInference class processes the image and sends it to LLaVA
4. LLaVA generates a text response describing the image
5. The ModelInference class parses the response to extract structured data
6. Structured data is returned to the Streamlit interface for visualization

## LLaVA Integration

### Model Loading

The application uses LLaVA's built-in model loading functions to load the model and its components:

```python
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path

# Load model components using LLaVA's loader
tokenizer, model, image_processor, context_len = load_pretrained_model(
    self.model_name,
    model_base=None,
    model_name=model_name
)
```

### Image Processing

Images are processed using the image processor provided by LLaVA:

```python
# Process the image
images = [processed_image]
image_sizes = [processed_image.size]
images_tensor = process_images(
    images,
    self.image_processor,
    self.model.config
).to(self.model.device, dtype=torch.float16)
```

### Response Generation

The application generates responses using LLaVA's conversation templates and generation function:

```python
# Generate response with the model
with torch.inference_mode():
    output_ids = self.model.generate(
        input_ids,
        images=images_tensor,
        image_sizes=image_sizes,
        do_sample=True,
        temperature=0.2,
        top_p=0.7,
        max_new_tokens=512,
        use_cache=True,
    )
```

## Key Design Decisions

1. **Model Selection**: We chose LLaVA-v1.5-7b because of its strong performance on image understanding tasks, English language output, and reasonable computational requirements.

2. **Fallback Mechanism**: For systems without GPU support or when model inference fails, we implemented a fallback to mock data to ensure the application remains functional for demonstration purposes.

3. **Modular Architecture**: The application is structured with clear separation between the UI (Streamlit), model interface (inference.py), and the underlying model (LLaVA), making it easy to swap out components or modify functionality.

4. **Response Parsing**: Instead of using the raw model output, we parse the responses using regular expressions to extract structured data that can be easily visualized in the UI.

5. **Prompt Engineering**: We carefully designed the prompts for each analysis task to guide the model in generating consistent, structured responses that can be parsed reliably.

## Installation and Deployment

### Dependencies

The application requires the following key dependencies:

- Streamlit for the UI
- PyTorch for model inference
- LLaVA for vision-language capabilities
- Plotly for data visualization

### Deployment Considerations

- **Hardware Requirements**: A CUDA-compatible GPU with at least 12GB of VRAM is recommended for optimal performance.
- **Memory Usage**: The LLaVA model requires approximately 8GB of VRAM during inference.
- **Scaling**: For production deployments, consider using a load balancer with multiple inference servers to handle concurrent requests.

## Future Enhancements

1. **Additional Analysis Modules**: Add modules for analyzing store layout, product placement, and customer flow patterns.
2. **Fine-tuning**: Fine-tune the LLaVA model on retail-specific images to improve accuracy.
3. **Historical Data**: Implement a database to store historical analysis results and track trends over time.
4. **Automation**: Add support for automated analysis of video feeds or scheduled image capture and analysis.
5. **API Integration**: Develop an API to allow integration with other retail management systems. 