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