# Store Insights AI

A Streamlit application for analyzing retail store images to gain insights about customer demographics and queue management using the LLaVA vision-language model.

## Features

- Gender demographics analysis: Count men and women in store images
- Queue management analysis: Analyze checkout counters and provide recommendations
- Dashboard with visualizations of key metrics
- Support for both real-time image uploads and sample images

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for model inference)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Store-Insights-AI.git
cd Store-Insights-AI
```

2. Clone the LLaVA repository into the LLaVA directory:
```bash
# Clone LLaVA into the LLaVA directory
git clone https://github.com/haotian-liu/LLaVA.git LLaVA
```

3. Create and activate a virtual environment (optional but recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Install LLaVA from the local repository:
```bash
cd LLaVA
pip install -e .
cd ..
```

## Usage

Run the Streamlit application:

```bash
streamlit run app/app.py
```

For systems with limited GPU memory, you can run with the `--small-model` flag:

```bash
streamlit run app/app.py -- --small-model
```

## How It Works

This application uses the LLaVA vision-language model (v1.5-7b) to analyze retail store images and extract useful insights. The model is capable of:

1. Counting the number of men and women in the image
2. Identifying products that customers are looking at
3. Providing insights about customer behavior and preferences
4. Analyzing checkout counters and queue management

If a CUDA-compatible GPU is not available, the application falls back to mock data for demonstration purposes.

## Project Structure

```
Store-Insights-AI/
├── app/                        # Main application code
│   ├── app.py                  # Streamlit application entry point
│   └── pages/                  # Individual page modules
│       ├── gender_demographics.py
│       └── queue_management.py
├── model/                      # Model interface code
│   └── inference.py            # Interface to LLaVA model
├── LLaVA/                      # LLaVA model repository
├── samples/                    # Sample images for testing
├── docs/                       # Documentation
└── README.md                   # This file
```

## License

This project is licensed under the MIT License. 