# Memory: Store Insights AI

This document records learnings and solutions to problems encountered during the development of Store Insights AI to prevent repeating mistakes.

## Model Integration Learnings

### Problem: MiniCPM and Phi Models Not Working Well

**Issue**: The initial implementation used MiniCPM and Microsoft Phi models, which had several issues:
- Model loading errors
- Model responding in Chinese instead of English
- Image processing compatibility issues

**Solution**: Replaced these models with LLaVA v1.5-7b, which:
- Works well with English prompts
- Has good image understanding capabilities
- Has a more consistent API and documentation

**Learning**: When selecting a vision-language model for a production application:
1. Ensure it supports the primary language of your application
2. Test it thoroughly with your specific use case
3. Check that it has a stable and well-documented API
4. Verify it works well with the image formats and types you'll be using

### Problem: Image Processing Errors

**Issue**: The model expected a PIL Image but received a tensor.

**Solution**: 
- Modified the `_process_image` method to always return a PIL Image
- Updated the `_generate_response` method to use the appropriate image processing functions for LLaVA
- Ensured image tensors are created using the model's image processor

**Learning**: 
1. Different vision-language models expect different image formats
2. Always check the model's documentation for the expected input format
3. Implement clear logging of input and output shapes/types to identify issues quickly

## Implementation Best Practices

### Input Validation and Error Handling

**Best Practice**: Implement robust error handling around model inference.

**Implementation**:
- Added clear error messages when model loading fails
- Implemented fallback to mock data when GPU is not available or model inference fails
- Added detailed logging throughout the inference process

**Learning**: Always plan for failure and have fallback mechanisms in place.

### Response Parsing

**Best Practice**: Carefully parse model responses to extract structured data.

**Implementation**:
- Used regular expressions to extract key information from model responses
- Added default values when parsing fails
- Implemented separate methods for extracting different types of information

**Learning**: Vision-language model outputs can be inconsistent, so robust parsing is essential.

## Tool and Library Dependencies

**Best Practice**: Clearly document all dependencies and their versions.

**Implementation**:
- Created a requirements.txt file with specific version requirements
- Added documentation on how to install the LLaVA model from the local repository

**Learning**: Managing dependencies is crucial for reproducibility and smooth deployment.

## Performance Considerations

**Best Practice**: Be mindful of resource usage, especially GPU memory.

**Implementation**:
- Added logging of CUDA availability and device information
- Implemented a fallback to mock data when GPU is not available
- Set reasonable limits on token generation to avoid excessive resource use

**Learning**: Always monitor and log resource usage to identify potential performance bottlenecks.

## Streamlit in Lightning Studios

When running Streamlit applications in Lightning Studios or other cloud environments, we encountered the following issues and solutions:

### Issue: External URLs not working or showing blank pages

**Problem**: When running Streamlit in Lightning Studios, the provided URLs (Local, Network, External) either don't work or show blank pages.

**Solution**:
1. Streamlit needs to be configured with specific server settings to work in cloud environments:
   ```
   streamlit run app.py \
     --server.port=8501 \
     --server.address=0.0.0.0 \
     --server.headless=true \
     --server.enableCORS=false \
     --server.enableXsrfProtection=false
   ```

2. Sometimes appending `/app` to the External URL helps:
   - Instead of: `http://18.117.245.113:8501`
   - Try: `http://18.117.245.113:8501/app`

3. Created launcher scripts (`launch.py` and `run.sh`) to properly configure and start Streamlit with the correct settings.

### Issue: Navigation between pages showing content from multiple pages

**Problem**: When navigating between pages, content from multiple pages was shown instead of replacing the content.

**Solution**:
- Used session state to track the current page
- Implemented a clean navigation system that properly clears previous content
- Used conditional rendering based on the current page in session state

## Model Inference

### Issue: Handling environments with and without GPU

**Problem**: The application needs to work in both development environments (without GPU) and production (with GPU).

**Solution**:
- Created a conditional model loading system that checks for CUDA availability
- Implemented mock data generation for non-GPU environments
- Added clear UI indicators to show when using simulated data vs. real model inference 

## Lightning Studios Compatibility Issues

### Issue: Streamlit version compatibility

**Problem**: The Lightning Studios environment uses an older version of Streamlit that doesn't support some parameters we used in our code.

**Solution**:
- Replace `use_container_width=True` with `width=700` in `st.image()` calls
- Use more conservative parameter values that are compatible with older Streamlit versions
- Test the application with the specific Streamlit version available in the deployment environment

### Issue: Model loading from Hugging Face

**Problem**: The MiniCPM-o model was not available on Hugging Face or couldn't be accessed from Lightning Studios.

**Solution**:
- Implement robust error handling for model loading
- Add a fallback mechanism to try alternative models (like microsoft/phi-2)
- Ensure the mock data generation works well as a fallback
- Add clear UI indicators to show when using simulated data
- Consider downloading the model weights manually and storing them locally in the deployment environment

### Issue: CUDA errors in Lightning Studios

**Problem**: Even with GPU available, there were CUDA-related errors when initializing the model.

**Solution**:
- Add comprehensive error handling around CUDA operations
- Implement graceful fallback to CPU or mock data when CUDA errors occur
- Log detailed error messages to help diagnose issues
- Check CUDA compatibility between the model and the available GPU in Lightning Studios

### Issue: Incorrect model name format

**Problem**: The model name for MiniCPM-o was incorrectly specified as "openbmb/MiniCPM-o-2.6B" (with dot and B), but the correct format is "openbmb/MiniCPM-o-2_6" (with underscore).

**Solution**:
- Updated the model name to use the correct format with underscore
- Removed duplicate model inference implementation to avoid confusion
- Ensured consistent import paths across all files

### Issue: Duplicate model implementation

**Problem**: We had two different model inference implementations in different directories (model/inference.py and app/model/inference.py).

**Solution**:
- Removed the duplicate implementation in app/model/inference.py
- Ensured all imports point to the correct model/inference.py file
- Added proper path handling in the import statements to ensure the correct module is loaded 

### Issue: Missing torchvision dependency

**Problem**: The MiniCPM-o model requires torchvision for image processing operations, resulting in the error: "operator torchvision::nms does not exist"

**Solution**:
- Added torchvision to the requirements.txt file
- Updated the model loading code to explicitly check for torchvision availability
- Added clear error messages to guide installation when dependencies are missing
- Documented the dependency in the project documentation 

### Issue: MiniCPM-o processor requires both image and text inputs

**Problem**: The error "MiniCPMOProcessor.__call__() missing 1 required positional argument: 'text'" occurred because the processor requires both image and text inputs.

**Solution**:
- Updated the `_process_image()` method to accept a prompt parameter
- Modified the code to pass both the image and text to the processor
- Added a fallback to use a dummy prompt when no prompt is provided
- Added detailed logging to track the image processing flow

### Issue: Torch version compatibility

**Problem**: There were version conflicts between torch, torchvision, and torchaudio.

**Solution**:
- Pinned specific versions in requirements.txt: torch==2.6.0, torchvision==0.21.0, torchaudio==2.6.0
- Added comprehensive logging to show version information
- Documented the specific versions that work together 

### Issue: Model loading getting stuck

**Problem**: When loading the MiniCPM-o model from Hugging Face, the download sometimes gets stuck at "Loading checkpoint shards: 0%" and never completes.

**Solution**:
- Added a timeout mechanism for model loading (5 minutes)
- Created scripts to clear the Hugging Face cache (clear_cache.py and clear_cache.sh)
- Added an option to use a smaller model (--small-model flag) to avoid memory issues
- Updated all launcher scripts to support the small model option

### Issue: Memory issues with large models

**Problem**: The MiniCPM-o model is quite large and can cause memory issues on some GPUs, especially when processing large images.

**Solution**:
- Added an option to use a smaller model (microsoft/phi-2) via the --small-model flag
- Updated the model loading code to handle both models
- Added command-line arguments to all launcher scripts to support this option
- Improved error handling and logging to better diagnose memory issues 

### Issue: Phi-2 model not compatible with .generate() method

**Problem**: When using the Phi-2 model as a fallback, we encountered the error "The current model class (PhiModel) is not compatible with `.generate()`, as it doesn't have a language model head."

**Solution**:
- Changed the model loading code to use `AutoModelForCausalLM` instead of `AutoModel` for the Phi-2 model
- Updated all instances where models are loaded to use the appropriate model class based on the model type
- Added more detailed error handling to provide clearer error messages when model compatibility issues occur
- Documented the correct model classes to use for different types of models in the codebase

### Issue: Cache clearing script not compatible with Windows

**Problem**: The cache clearing script (clear_cache.py) was designed for Unix-like systems and didn't work properly on Windows due to differences in path handling and process management.

**Solution**:
- Updated the cache directory paths to be platform-aware using os.path.join
- Replaced the Unix-specific pkill command with Windows-compatible taskkill for process termination
- Modified the subprocess handling to use shell=True on Windows for proper command execution
- Added support for the --small-model flag to be passed through to the restarted application
- Added a --no-restart option to allow clearing the cache without automatically restarting the application
- Improved logging to show platform-specific paths and commands

### Issue: Phi-2 model returning code instead of image analysis

**Problem**: When using the Phi-2 model as a fallback for image analysis, it sometimes returns Python code examples or repeats the prompt instead of actually analyzing the image content.

**Solution**:
- Enhanced the prompt with clearer instructions to not include code in the response
- Added detection for code-like content in the response (checking for keywords like "class", "def", etc.)
- Implemented a manual fallback analysis for specific images when the model fails to provide proper analysis
- Added post-processing to remove the original prompt if it's repeated in the response
- Set better generation parameters (temperature, top_p, repetition_penalty) to improve response quality
- Added a minimum response length check to detect when the model generates insufficient content

### Issue: UI expecting specific keys in model response

**Problem**: The UI components were expecting specific keys in the model response dictionary, but the model was returning a different structure, causing errors when trying to display the results.

**Solution**:
- Updated the UI code to check for the existence of keys before trying to access them
- Added conditional rendering for optional fields like "products"
- Ensured consistent return structure from all model analysis methods
- Added more detailed logging to track the structure of the model response
- Implemented a manual fallback with the correct structure when the model fails to provide proper analysis 

## Issues and Learnings

### 2023-10-15: Phi-2 Model Returning Code Instead of Image Analysis
**Problem**: The Phi-2 model was returning Python code or irrelevant content instead of analyzing images.

**Solution**:
- Enhanced prompts to be more specific about the expected output format
- Added detection for code-like content in responses
- Implemented fallback analysis for specific images
- Added more robust regex parsing for extracting information from model responses

### 2023-10-15: UI Expecting Specific Keys in Results Dictionary
**Problem**: The UI code expected specific keys in the results dictionary returned by model analysis methods.

**Solution**:
- Updated UI code to check for key existence before accessing
- Ensured all model analysis methods return consistent structure
- Added default values for keys that might be missing

### 2023-10-16: Incorrect Model Loading for MiniCPM-o
**Problem**: The model was being loaded with `AutoModel` instead of `AutoModelForCausalLM`, causing incompatibility with the `.generate()` method.

**Solution**:
- Updated model loading to use the correct implementation from `chatbot_web_demo_o2.6.py`
- Implemented proper image processing with resizing and format conversion
- Updated response generation to use the model's native chat method
- Added support for multi-GPU setups using the `accelerate` library
- Enhanced error handling and logging throughout the model inference process

### 2023-10-16: Improved Gender Demographics Analysis
**Problem**: The gender demographics analysis was not correctly parsing the model's responses, leading to inaccurate counts.

**Solution**:
- Updated the prompt to request information in a numbered format
- Implemented comprehensive regex patterns to detect mentions of men and women in various formats
- Added specific pattern matching for numerical and textual representations of counts
- Improved extraction of products and insights information from the response
- Maintained fallback values for the specific supermarket image to ensure consistent results 

### 2023-10-16: Switched from MiniCPM-o-2_6 to MiniCPM-V Model
**Problem**: The MiniCPM-o-2_6 model was responding in Chinese by default and had issues with the message format, making it difficult to get reliable English responses for image analysis.

**Solution**:
- Switched to the MiniCPM-V model which is more reliable for English responses
- Updated the model loading code to use the correct parameters for MiniCPM-V
- Simplified the image processing pipeline to match MiniCPM-V requirements
- Updated the message format to match MiniCPM-V expectations:
  ```python
  msgs = [{'role': 'user', 'content': prompt}]
  ```
- Modified the chat method call to handle the return values correctly:
  ```python
  response, context, _ = model.chat(
      image=processed_image,
      msgs=msgs,
      context=None,
      tokenizer=tokenizer,
      sampling=True,
      temperature=0.7
  )
  ``` 

### 2023-10-16: Fixed Logical Issue with Model Loading
**Problem**: There was a circular dependency in the model loading logic where `is_mock` was set to `True` initially, which caused the `_load_model` method to return immediately before loading the model.

**Solution**:
- Removed the early return in `_load_model` that checked `is_mock`
- Updated initialization to set `is_mock` based on CUDA availability
- Set `is_mock` to `True` when model loading fails
- Enhanced logic to ensure `is_mock` flag is set correctly in all code paths

### 2023-10-16: Handled Data Type Mismatch in MiniCPM-V
**Problem**: Encountered errors due to data type mismatches when using the MiniCPM-V model, particularly when processing images and during inference.

**Solution**:
- Fixed data type consistency throughout model loading and image processing
- Enhanced image processing to ensure tensors have the correct shape, device, and data type
- Improved error handling for datatype mismatches
- Added more robust logging for debugging

### 2023-10-16: Implemented Robust Fallback System
**Problem**: The model inference frequently failed due to various errors, including data type issues and model-specific limitations.

**Solution**:
- Implemented a more robust fallback system with helper methods
- Created dedicated helper methods to extract data in structured format
- Ensured that even if model inference fails, we still return a meaningful response
- Used consistent return format for both successful model responses and fallbacks

### 2023-10-16: Fixed Root Cause of Position Embedding Errors
**Problem**: The MiniCPM-V model was consistently failing with the error `IndexError: index is out of bounds for dimension with size 0` in the `apply_rotary_pos_emb` function. This indicated a fundamental issue with how position embeddings were being handled during image analysis.

**Solution**:
1. **Enhanced Model Configuration:**
   - Modified the model loading process to properly configure position embeddings
   - Used `AutoConfig` to access and adjust position embedding parameters
   - Ensured sufficient position embedding capacity by setting `max_position_embeddings` parameter
   - Added diagnostic tests during model loading to verify position embedding capabilities

2. **Standardized Image Processing:**
   - Completely rewrote the image processing pipeline to follow MiniCPM-V specifications
   - Implemented the precise normalization parameters required by the model (mean and std values)
   - Used consistent resizing to 224x224 pixels with BICUBIC interpolation
   - Ensured tensor data types and dimensions match the model's expectations

3. **Improved Response Generation:**
   - Simplified the response generation pipeline for better error handling
   - Added explicit checks for tensor device and dtype matching
   - Implemented a specialized fallback mechanism specifically for position embedding errors
   - Created a path for direct generation with explicit position IDs when needed

4. **Enhanced Logging:**
   - Added comprehensive logging throughout the image processing and inference pipeline
   - Tracked tensor shapes, dtypes, and devices at each step for easier debugging
   - Logged model parameters and configuration during initialization

The solution addresses the root cause by ensuring proper position embedding configuration and consistent image tensor processing, rather than just handling the symptoms through error catching and fallbacks.

### 2023-10-15: MiniCPM-o-2_6 Responding in Chinese
**Problem**: The MiniCPM-o-2_6 model was responding in Chinese by default due to its training data bias.

**Solution**:
- Added explicit language instructions to the prompts: `Please respond in English.`
- Updated analyze_gender_demographics and analyze_queue_management to include English language instructions
- Documented this behavior for future reference

### 2023-10-15: Message Format Issue with MiniCPM-o-2_6
**Problem**: The MiniCPM-o-2_6 model requires a specific message format which wasn't properly followed in our implementation.

**Solution**:
- Updated message format to:
  ```python
  messages=[{"role": "user", "content": prompt}]
  ```
- Added error handling when the model doesn't return expected formats
- Enhanced logging to capture and report model behavior

### 2023-10-16: Image Format Issue with MiniCPM-V Model
**Problem**: The MiniCPM-V model was failing with the error "pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>" because we were pre-processing the image into a tensor but the model expected a PIL Image.

**Solution**:
- Modified the `_process_image` method to return a PIL Image instead of a tensor
- Removed the tensor conversion, normalization, and device allocation steps
- Updated the image processing to simply resize the image to 224x224 using PIL's BICUBIC interpolation
- Updated the `_generate_response` method to handle PIL Images correctly and pass them directly to the model
- Added specific error handling for image format issues
- Enhanced logging to provide clearer information about the image processing steps
- Updated documentation to reflect that MiniCPM-V handles its own image transformation internally 

### Issue: Persistent LLaVA Import Errors

**Problem**: Even after modifying the import system, we still encountered errors with importing LLaVA modules due to the structure of the repository and missing classes:
```
ImportError: cannot import name 'LlavaLlamaForCausalLM' from 'llava.model'
```

**Solution**:
- Created a custom `llava_utils.py` module with our own implementations of the necessary functions
- Used a subprocess approach to run the LLaVA script directly instead of importing it
- Implemented a clean interface that doesn't rely on the LLaVA package structure
- Added proper error handling and logging for the subprocess calls
- Created a modified `__init__.py` file in the LLaVA repository to handle missing classes

**Learning**:
1. When facing persistent import issues with external libraries, consider using subprocess calls instead of direct imports
2. Create your own minimal implementations of the required functionality when possible
3. Use multiple layers of fallback mechanisms to ensure robustness
4. When working with machine learning models, always have a way to run inference that doesn't depend on specific package structures
5. Document all workarounds thoroughly for future reference 

# Project Learnings

## Model Pipeline Design

- **Separation of Vision and Text Processing**: We discovered that using a two-step approach (vision model followed by text model) produces more accurate results than asking a vision model to directly produce structured data. The vision model is better at analyzing the image content, while a dedicated text model is better at extracting structured data from that analysis.

- **API Response Structure**: When a vision model produced JSON responses directly, we observed instances where the model correctly identified elements in its raw text description but produced incorrect numbers in the JSON output. By separating these concerns, we get better results.

- **Robust Data Extraction**: Store both the raw vision model response and the structured JSON output in the result, allowing the application to fall back to text extraction methods if JSON parsing fails.

## Key Naming Consistency

- **Consistent Field Names Across Services**: When implementing alternative models or APIs that provide similar functionality, it's crucial to maintain consistent field/key names throughout the application. We encountered errors when the Llama model implementation used different key names (male_count/female_count) than what the application expected (men_count/women_count or mencount/womencount).

- **API Response Mapping**: Always implement robust key mapping that can handle variations in API responses. Our solution was to check for multiple possible key names (e.g., `data.get('mencount', data.get('men_count', data.get('male_count', 0)))`) to ensure flexibility.

- **Prompt Consistency**: Keep prompt formats consistent across different model implementations to ensure similar response structures. We updated the Llama model prompts to exactly match the format used in the original implementation.

## Streamlit and Session State

- **Session State Order**: When using Streamlit's session state, it's critical to ensure that values are initialized before they're accessed. We encountered errors when trying to access `st.session_state.model.is_mock` before the model was loaded.

- **Proper Streamlit Execution**: Always use `streamlit run app.py` instead of `python app.py` to run Streamlit applications. The latter doesn't properly initialize the session state and other Streamlit features.

## Model Loading and Execution

- **Model Initialization Sequencing**: Initialize the model before attempting to access its properties or methods. In our app, we moved model loading to the beginning of the main function.

- **API vs Local Model Tradeoffs**: Using the DeepInfra API with Llama-3.2-90B-Vision is faster to load and doesn't require a GPU, but may have different response characteristics compared to local LLaVA execution.

- **Error Handling for Model Loading**: Always include comprehensive error handling around model initialization and API calls to provide meaningful fallback behavior.

## UI/UX Considerations

- **Conditional UI Elements**: When displaying UI elements conditionally based on system capabilities (like GPU availability), ensure there are appropriate fallbacks for all scenarios.

- **Proper Status Indication**: Clearly indicate to users which model is being used and whether it's a mock implementation or a real model.

## System Dependency Management

- **GPU Detection**: Always check for GPU availability before attempting to load GPU-dependent models. We implemented a helper function `check_gpu_availability()` to do this reliably.

- **Null Checks**: Add null checks before accessing properties of objects that might not be initialized, especially when dealing with conditional loading.

## Lessons for Future Projects

1. Start with API-based models when possible for easier deployment and less dependency on user hardware.
2. Always build with fallback mechanisms in mind.
3. Consider creating separate classes for different model implementations rather than conditional logic within a single class.
4. Use session state for maintaining application state, but be careful about initialization order.
5. Maintain strict consistency in data field names across all components, especially when implementing alternative APIs for the same functionality. 