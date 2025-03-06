# Store Insights AI

A computer vision tool for retail analytics that uses the LLaVA multimodal model to extract insights from store images, including gender demographics and queue management.

## Project Structure

- `model/inference.py`: Contains the core `ModelInference` class for running image analysis
- `test_llava_hf.py`: Test script for running inference using the LLaVA model (requires GPU)
- `test_extraction_logic.py`: Test script for testing just the text extraction logic without running the model
- `test_specific_response.py`: Simplified test script for testing extraction logic on a specific model response

## Requirements

The following versions were tested and confirmed working for the full model:

```
tokenizers==0.20.3
accelerate==1.4.0
torch==2.1.2+cu121
torchvision==0.16.2+cu121
transformers==4.45.2
timm==1.0.15
bitsandbytes==0.45.3
pillow>=10.0.0
numpy>=1.24.0
tqdm>=4.66.0
```

For just running the extraction logic tests (no model inference), you only need Python with standard libraries.

## Setup

1. For the full model, install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure you have a GPU with CUDA support for optimal performance when running the full model.

## Usage

### Running the Full Model (Requires GPU)

The full model requires a GPU for efficient inference:

```bash
python test_llava_hf.py --mode both --image /path/to/your/image.jpg
```

Arguments:
- `--mode`: Choose which implementations to test: `direct`, `module`, or `both`
- `--image`: Path to the image you want to analyze

### Testing Extraction Logic (No GPU Required)

To test just the text processing logic without running the model inference, you can use either of these test scripts:

#### Option 1: Test with a specific model response

```bash
python test_specific_response.py
```

This tests extraction on the exact response: "In the image, there are two men and two women. They are looking at various products, including bottles and cans, which are displayed in the store."

#### Option 2: Test with custom text or sample texts

```bash
# Test gender extraction with custom text
python test_extraction_logic.py --type gender --text "In the image, I can see three men and two women looking at electronics."

# Test queue extraction with custom text
python test_extraction_logic.py --type queue --text "There are 2 checkout counters with 5 customers waiting in line."

# Test with multiple sample texts
python test_extraction_logic.py --type both --samples
```

Arguments:
- `--type`: Type of extraction to test: `gender`, `queue`, or `both`
- `--text`: Custom text to test extraction on
- `--samples`: Run tests with multiple predefined sample texts

## Features

### Gender Demographics Analysis

Analyzes retail store images to extract:
- Number of men and women
- Products they are looking at
- Additional insights

### Queue Management Analysis

Analyzes retail store images to extract:
- Number of open checkout counters
- Number of customers in queue
- Queue efficiency
- Wait time estimates
- Recommendations

## Model Implementation

The implementation uses the LLaVA 1.5 7B model from Hugging Face, with the following optimizations:
- 4-bit quantization for lower memory usage
- GPU acceleration with CUDA
- Simplified prompt structure for reliable responses 