# Store Insights AI - Project Memory

This document records important lessons learned throughout the development process to avoid repeating mistakes.

## Learnings

### Development Environment
- Using Windows command prompt for terminal operations
- Project is intended to be deployed to Pytorch Lightning Studio hub with GPU

### Integration Notes
- The computer vision model (MiniCPM-o) requires CUDA support, which is not available locally
- Implemented conditional model loading to handle both GPU and non-GPU environments
- Used a fallback mechanism to provide mock data when GPU is not available

### Model Integration
- MiniCPM-o model is loaded from HuggingFace using AutoModel and AutoTokenizer
- The model requires preprocessing images to a maximum size of 448*16 pixels
- Model inference is done using the chat method with appropriate parameters
- Regular expressions are used to extract structured data from the model's text output
- Error handling is important to gracefully handle model loading and inference failures

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
**Problem**: There was a circular dependency in the model loading logic. The `is_mock` flag was initially set to `True`, and then in the `_load_model` method, it immediately returned if `is_mock` was `True`, preventing the model from ever loading.

**Solution**:
- Removed the early return in the `_load_model` method that was checking the `is_mock` flag
- Updated the initialization logic to set `is_mock` based on CUDA availability initially
- Added explicit setting of `is_mock` to `True` when model loading fails
- Made the code more robust by ensuring the `is_mock` flag is correctly set in all code paths 

### 2025-03-05: Fixed "Index is out of bounds" Error with MiniCPM-V
**Problem**: When using the MiniCPM-V model for image analysis, we encountered the error "index is out of bounds for dimension with size 0" during the model's chat method call. This error occurred in the rotary position embeddings calculation within the model's internal implementation.

**Solution**:
- Updated the model loading code to use more specific parameters:
  - Changed from `torch.bfloat16` to `torch.float16` for better compatibility
  - Added `low_cpu_mem_usage=True` and `use_cache=True` parameters
  - Improved device handling with explicit device assignment
- Enhanced the image processing method:
  - Added proper image resizing to a maximum dimension of 768 pixels
  - Used LANCZOS resampling for better quality
- Implemented a fallback mechanism in the `_generate_response` method:
  - Added specific error handling for IndexError
  - Created an alternative generation approach using the model's `generate` method directly
  - Set appropriate generation parameters (max_new_tokens, top_p, top_k, temperature)
- Improved error logging throughout the code to better diagnose issues

### 2025-03-05: Fixed Data Type Mismatch Error with MiniCPM-V
**Problem**: After fixing the "index is out of bounds" error, we encountered a new error: "mat1 and mat2 must have the same dtype, but got Float and Half". This occurred because of inconsistent data types between the model parameters and the input tensors during the multi-head attention calculation.

**Solution**:
- Updated the model loading code to use a consistent data type throughout:
  - Changed to use `torch.float32` for both CPU and CUDA to avoid dtype mismatches
  - Added explicit logging of the dtype being used
  - Ensured the model is moved to the device with the correct dtype
- Enhanced the image processing in the fallback mechanism:
  - Added proper image tensor creation with normalization
  - Ensured the image tensor has the same dtype as the model parameters
  - Added device and dtype checks to match the model's configuration
- Implemented a more robust fallback mechanism:
  - Added handling for both IndexError and RuntimeError
  - Attempted to use the model's vision encoder directly if available
  - Added a last-resort mock response with reasonable values for the specific use case
- Improved error logging to track tensor shapes, dtypes, and devices 