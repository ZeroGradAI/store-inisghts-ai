# Store Insights AI

An intelligent assistant for retail store management that uses vision models to analyze store images and provide insights on customer demographics and queue management.

## Features

- **Gender Demographics Analysis**: Analyze the gender distribution of customers in the store.
- **Queue Management Analysis**: Get insights on checkout counter efficiency and identify overcrowded areas.
- **Model Selection**: Choose between two powerful vision AI models:
  - **Llama-3.2-90B-Vision** (via DeepInfra API): Faster, API-based vision model (default)
  - **LLaVA-1.5-7B**: Local vision-language model (requires GPU)

## Setup

### Prerequisites

- Python 3.8 or later
- [Conda](https://docs.conda.io/en/latest/) (recommended for environment management)
- GPU with CUDA support (optional, for LLaVA model)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Store-Insights-AI.git
   cd Store-Insights-AI
   ```

2. Create and activate a conda environment:
   ```
   conda create -n store-insights python=3.8
   conda activate store-insights
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit app:
   ```
   streamlit run app/app.py
   ```

2. Access the app in your browser at http://localhost:8501

3. Choose your preferred model in the sidebar:
   - **Llama** (default): Uses DeepInfra's Llama-3.2-90B-Vision-Instruct model via API
   - **Llava**: Uses local LLaVA-1.5-7B model (requires GPU)

4. Upload images of your store to get real-time insights

## Components

- `app/app.py`: Main Streamlit application
- `app/pages/`: Individual pages for different analyses
- `model/inference.py`: LLaVA model inference
- `model/inference_llama.py`: Llama model inference via DeepInfra API
- `samples/`: Sample images for testing

## Notes

- If your system doesn't have a GPU, the application will automatically use the Llama model via DeepInfra API.
- For the best experience with the LLaVA model, a GPU with at least 8GB of VRAM is recommended.

## License

[MIT License](LICENSE) 