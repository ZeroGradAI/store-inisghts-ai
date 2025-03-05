# Store Insights AI - Feature Design

## Project Overview
Store Insights AI is an application designed for store managers to gain valuable insights about their stores using computer vision technology. The system analyzes images from store cameras to extract demographic information and queue statistics.

## Modules and Features

### 1. Main Dashboard
- Displays a summary of insights from both the Gender Demography and Queue Management modules
- Shows key metrics and visualizations of store performance
- Provides quick access to both modules

### 2. Gender Demography Module
- Allows users to upload images of customers browsing products
- Uses computer vision to:
  - Count the number of men and women in the image
  - Analyze what products customers are looking at
  - Generate insights about customer demographics and interests
- Displays results with visual indicators and metrics

### 3. Queue Management Module
- Allows users to upload images of checkout counters (top view)
- Uses computer vision to:
  - Count total billing counters
  - Identify which counters are open/closed
  - Detect overcrowded counters
  - Suggest opening new counters when needed
- Displays results with visual indicators and counter status information

## Technology Stack
- Frontend: Streamlit (Python web app framework)
- Image Analysis: MiniCPM-o model for computer vision and insights
- Data Visualization: Streamlit native components and Plotly

## User Flow
1. User accesses the main dashboard
2. User navigates to either Gender Demography or Queue Management module
3. User uploads an image for analysis
4. System processes the image and displays results
5. Main dashboard updates with latest insights 