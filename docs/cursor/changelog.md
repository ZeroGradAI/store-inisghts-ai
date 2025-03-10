# Changelog: Store Insights AI

## [Unreleased]

### Added
- Integration with LLaVA v1.5-7b model for image analysis
- Requirements.txt with updated dependencies
- Comprehensive README.md with installation and usage instructions
- Improved documentation including feature design and current state
- Python path fix to properly import LLaVA modules
- LLaVA directory README and setup instructions
- Temporary file handling for saving processed images
- Custom llava_utils.py module with subprocess-based model execution
- Modified LLaVA __init__.py to handle missing classes gracefully
- Added DeepInfra API integration using Llama-3.2-90B-Vision-Instruct model
- Created `inference_llama.py` for API-based model inference
- Implemented model selection in the Streamlit app between Llama and Llava models
- Added automatic fallback to Llama model if GPU is not available
- Added model information display in the sidebar
- Updated README.md with new model options and instructions

### Changed
- Replaced MiniCPM and Phi models with LLaVA model in inference.py
- Updated image processing pipeline to work with LLaVA
- Modified response generation and parsing for LLaVA's output format
- Changed model references throughout the codebase
- Updated main README with detailed LLaVA setup instructions
- Rewritten model loading and inference to use the eval_model approach instead of directly importing model classes
- Enhanced error handling and fallback mechanisms
- Switched to subprocess-based model execution to avoid import issues
- Modified app.py to use session state for model storage
- Updated gender_demographics.py and queue_management.py to use the model from session state
- Improved error handling in model initialization
- Moved model loading to the beginning of the main function to prevent null reference errors
- Reorganized sidebar to include model selection options

### Removed
- MiniCPM and Phi model-specific code
- Unused dependencies related to previous models

## [0.1.0] - Initial Version

### Added
- Basic Streamlit application structure
- Dashboard for displaying metrics and insights
- Gender Demographics module for analyzing customer demographics
- Queue Management module for analyzing checkout counters
- Mock data fallback for systems without GPU 