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