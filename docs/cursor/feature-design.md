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

# Feature Design: Model Selection and Integration

## Overview

This feature allows users to choose between two vision AI models for analyzing store images:

1. **Llama-3.2-90B-Vision-Instruct** (via DeepInfra API)
2. **LLaVA-1.5-7B** (local model, requires GPU)

The system automatically defaults to the Llama model if no GPU is available, ensuring the application works across different hardware configurations.

## Architecture

### Components

1. **Model Interface Classes**:
   - `ModelInference` (in inference.py): Local LLaVA model implementation
   - `LlamaModelInference` (in inference_llama.py): DeepInfra API-based implementation

2. **UI Components**:
   - Model selection dropdown in the sidebar
   - Model status indicator showing which model is active

3. **Session State Management**:
   - `st.session_state.selected_model`: Tracks which model is selected ('llama' or 'llava')
   - `st.session_state.model`: Stores the active model instance

### Data Flow

1. On application startup:
   - Check for GPU availability using `check_gpu_availability()`
   - Initialize session state with default model ('llama')
   - If GPU is not available, hide the model selection UI and use 'llama' by default

2. When user changes model selection:
   - Update `st.session_state.selected_model`
   - Set `st.session_state.model` to None to trigger reloading
   - Reload the appropriate model instance

3. For analysis operations:
   - All analysis functions (gender demographics, queue management) use the model from session state
   - Each function checks if the model is loaded before proceeding

## Implementation Details

### API Integration (inference_llama.py)

- Uses the OpenAI-compatible API provided by DeepInfra
- Handles image conversion to base64 format for API consumption
- Extracts structured data from model responses
- Provides fallback mechanisms for error cases

### Model Selection UI

```python
# Only show model selection if GPU is available, otherwise default to Llama
if has_gpu:
    model_options = ['llama', 'llava']
    model_selection = st.selectbox(
        "Select Vision Model",
        options=model_options,
        index=0,  # Default to Llama
        help="Llama is faster and uses less memory. Llava may provide more accurate results but requires a GPU."
    )
    
    if model_selection != st.session_state.selected_model:
        st.session_state.selected_model = model_selection
        st.session_state.model = None  # Reset the model so it will be reloaded
        st.rerun()  # Rerun the app to load the new model
else:
    st.info("GPU not available. Using Llama model.")
    st.session_state.selected_model = 'llama'
```

### Model Loading Logic

```python
# Load the selected model if it hasn't been loaded yet
if st.session_state.model is None:
    with st.spinner(f"Loading {st.session_state.selected_model.upper()} model..."):
        if st.session_state.selected_model == 'llama':
            st.session_state.model = get_llama_model()
        else:  # llava
            st.session_state.model = get_llava_model(use_small_model=args.small_model)
```

## Considerations

1. **Error Handling**:
   - Comprehensive error checking for API failures
   - Fallback to mock data if model initialization fails
   - Null checks before accessing model attributes

2. **User Experience**:
   - Clear indication of which model is active
   - Informative UI elements explaining the trade-offs between models
   - Hiding unavailable options to prevent user confusion

3. **Performance**:
   - Lazy loading of models to minimize startup time
   - Using session state to avoid reloading the model on every page navigation
   - Progress indicators during model loading and inference

4. **Extensibility**:
   - Common interface between different model implementations
   - Consistent return format for analysis functions
   - Easy to add additional models in the future by following the same pattern 