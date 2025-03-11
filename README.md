# Store Insights AI

An intelligent assistant for retail store management that uses vision models to analyze store images and provide insights on customer demographics and queue management.

## Features

- **Gender Demographics Analysis**: Analyze the gender distribution of customers in the store.
- **Queue Management Analysis**: Get insights on checkout counter efficiency and identify overcrowded areas.
- **Model Selection**: Choose between three powerful vision AI models via DeepInfra API:
  - **Llama-3.2-11B-Vision-Instruct**: Default option - reliable and cost-effective
  - **Llama-3.2-90B-Vision-Instruct**: More powerful but potentially slower
  - **Microsoft Phi-4-Multimodal**: Alternative model with different capabilities

## Setup

### Prerequisites

- Python 3.8 or later
- [Conda](https://docs.conda.io/en/latest/) (recommended for environment management)
- DeepInfra API key

### Environment Variables

Create a `.env.local` file in the root directory with the following variables:

```
DEEPINFRA_API_KEY=your_api_key_here
```

You can also customize other settings:

```
LLAMA_VISION_MODEL_ID=meta-llama/Llama-3.2-11B-Vision-Instruct
LLAMA_VISION_MODEL_ID_90B=meta-llama/Llama-3.2-90B-Vision-Instruct
PHI_VISION_MODEL_ID=microsoft/Phi-4-multimodal-instruct
TEXT_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
DEFAULT_MODEL=llama
```

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
   - **Llama-3.2-11B** (default): Uses DeepInfra's Llama-3.2-11B-Vision-Instruct model
   - **Llama-3.2-90B**: Uses DeepInfra's Llama-3.2-90B-Vision-Instruct model
   - **Microsoft Phi-4**: Uses DeepInfra's Phi-4-multimodal-instruct model

4. Upload images of your store to get real-time insights

## Components

- `app/app.py`: Main Streamlit application
- `app/pages/`: Individual pages for different analyses
- `model/inference_llama.py`: API-based model inference for all models
- `samples/`: Sample images for testing
- `config.py`: Centralized configuration 

## Deployment

This application can be deployed to cloud platforms like Render and Vercel. For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

### Quick Deployment Links

- Deploy to Render: [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## Notes

- The Llama-3.2-11B model is set as default because it's reliable and cost-effective.
- When deploying, make sure to set the DEEPINFRA_API_KEY environment variable.

## License

[MIT License](LICENSE) 