# Deployment Guide

This guide explains how to deploy Store Insights AI to cloud platforms including Render and Vercel.

## Environment Variables

The following environment variables need to be set for proper functioning:

- `DEEPINFRA_API_KEY`: Your DeepInfra API key (required)
- `PHI_VISION_MODEL_ID`: Microsoft Phi-4 model ID (default: "microsoft/Phi-4-multimodal-instruct")
- `LLAMA_VISION_MODEL_ID`: Llama vision model ID (default: "meta-llama/Llama-3.2-90B-Vision-Instruct")
- `TEXT_MODEL_ID`: Text model ID for parsing (default: "meta-llama/Meta-Llama-3.1-8B-Instruct")
- `MAX_TOKENS`: Maximum tokens for model response (default: 32000)
- `DEFAULT_MODEL`: Default model to use, 'phi', 'llama', or 'llava' (default: 'phi')
- `USE_SMALL_MODEL`: Whether to use a smaller model, 'true' or 'false' (default: 'false')

## Deploying to Render

### Prerequisites

- A [Render](https://render.com/) account
- Your DeepInfra API key

### Steps

1. **Create a new Web Service**
   - Log in to your Render dashboard
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository

2. **Configure the service**
   - Name: `store-insights-ai` (or your preferred name)
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false`

3. **Set Environment Variables**
   - Add the following environment variables under the "Environment" section:
     - `DEEPINFRA_API_KEY`: Your DeepInfra API key
     - `DEFAULT_MODEL`: Set to 'phi' to use the Microsoft Phi-4 model (recommended)
     - Any other variables you want to customize

4. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application

5. **Access Your App**
   - Once deployed, you can access your app at the URL provided by Render

## Deploying to Vercel

### Prerequisites

- A [Vercel](https://vercel.com/) account
- Your DeepInfra API key

### Steps

1. **Install Vercel CLI (Optional, for local development)**
   ```bash
   npm install -g vercel
   ```

2. **Create a `vercel.json` file in your project root:**
   ```json
   {
     "builds": [
       {
         "src": "index.py",
         "use": "@vercel/python"
       }
     ],
     "routes": [
       {
         "src": "/(.*)",
         "dest": "index.py"
       }
     ],
     "env": {
       "PYTHONPATH": "."
     }
   }
   ```

3. **Create a simple entry point for Vercel (index.py in project root):**
   ```python
   # index.py
   import os
   import subprocess
   
   def start_app():
       os.environ["STREAMLIT_SERVER_PORT"] = "8501"
       os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
       os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
       subprocess.call(["streamlit", "run", "app/app.py"])
   
   if __name__ == "__main__":
       start_app()
   ```

4. **Deploy to Vercel:**
   - Using Vercel Dashboard:
     - Import your GitHub repository
     - Configure the project:
       - Build Command: `pip install -r requirements.txt`
       - Output Directory: `public`
       - Development Command: `python index.py`
   
   - Set environment variables in the Vercel dashboard:
     - `DEEPINFRA_API_KEY`: Your DeepInfra API key
     - `DEFAULT_MODEL`: Set to 'phi' to use the Microsoft Phi-4 model (recommended)
     - Any other variables you want to customize

5. **Deploy**
   - Click "Deploy"
   - Vercel will build and deploy your application

6. **Access Your App**
   - Once deployed, you can access your app at the URL provided by Vercel

## Model Selection Considerations

When deploying, consider which model is best for your use case:

1. **Microsoft Phi-4 (Default)**: Recommended for most deployments due to its:
   - Reliability (fewer safety filter rejections)
   - Cost-effectiveness (less expensive API calls)
   - No GPU requirements (works on any hosting tier)

2. **Llama-3.2-90B**: Alternative API option that:
   - May provide different quality results for some images
   - Has higher API usage costs
   - May occasionally trigger safety filters for store images

3. **LLaVA-1.5-7B**: Only suitable if your hosting provides GPU access:
   - Requires a GPU with 8GB+ VRAM
   - More costly to host due to GPU requirements
   - Provides local inference without API costs

## Using a Custom Domain

Both Render and Vercel allow you to use custom domains for your deployed applications:

1. **On Render**:
   - Go to your web service
   - Click on "Settings" > "Custom Domain"
   - Follow the instructions to add your domain

2. **On Vercel**:
   - Go to your project dashboard
   - Click on "Settings" > "Domains"
   - Follow the instructions to add your domain

## Troubleshooting

If your application is not working correctly after deployment:

1. **Check Logs**
   - Both platforms provide logs that can help diagnose issues
   - Look for error messages related to API keys or model loading

2. **Environment Variables**
   - Ensure all required environment variables are set
   - Verify your DeepInfra API key is correct

3. **Memory Limits**
   - If your app is crashing due to memory issues:
     - Set `USE_SMALL_MODEL=true` in environment variables
     - Consider upgrading to a higher tier on your hosting platform

4. **CORS Issues**
   - If you're experiencing CORS errors:
     - Make sure to set the proper Streamlit server settings
     - For Render: `--server.enableCORS false`
     - For Vercel: Set `STREAMLIT_SERVER_ENABLE_CORS=false` 