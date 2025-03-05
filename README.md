# Store Insights AI

A computer vision-based analytics application for store managers to gain insights from in-store camera images.

## Overview

Store Insights AI processes images from store cameras to provide valuable insights for store management. The application includes:

- **Gender Demography Analysis**: Count and analyze customers by gender and their product interests
- **Queue Management**: Monitor checkout counters, detect overcrowding, and recommend actions
- **Insights Dashboard**: Visualize and track store metrics over time

## Technology Stack

- **Frontend**: Streamlit
- **Computer Vision**: MiniCPM-o model
- **Deployment**: Pytorch Lightning Studio hub with GPU support

## Project Structure

```
Store Insights AI/
├── app/                  # Main application code
│   ├── app.py            # Main Streamlit application entry point
│   ├── pages/            # Individual module pages
│   ├── utils/            # Utility functions
│   └── components/       # Reusable UI components
├── data/                 # Sample data and images
├── model/                # Model integration code
│   └── inference.py      # MiniCPM-o model inference
├── docs/                 # Documentation
│   ├── cursor/           # Project tracking documents
│   └── demo_mvp.txt      # Original requirements
└── requirements.txt      # Python dependencies
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app/app.py
   ```

## Development

This project is designed to work with the MiniCPM-o model, which requires CUDA support. The application is structured to handle image uploads locally, but will perform model inference when deployed to a GPU environment through Pytorch Lightning Studio hub. 