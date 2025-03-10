# Assume openai>=1.0.0
import asyncio
from openai import OpenAI
from PIL import Image
import base64
import sys
import os

# Add the app directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key="UV0rOuIVezxXXYRsA7e4dz8PJYdojKeL",
    base_url="https://api.deepinfra.com/v1/openai",
)


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "store-image2.jpg"

# Getting the Base64 string
base64_image = encode_image(image_path)

async def get_chat_completion():
    return await asyncio.to_thread(openai.chat.completions.create,
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        # model="microsoft/Phi-4-multimodal-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """ 1. Estimate the approximate number of men and women customers shopping in the image.
                            2. Identify what products or store sections these customers appear to be browsing or interested in
                            3. Provide general insights about customer shopping patterns that might be useful for retail management

                            Please provide a detailed but objective description.""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=8000,
    )

# Define an async function to call get_chat_completion
async def main():
    chat_completion = await get_chat_completion()
    print(chat_completion.choices[0].message.content)
    print(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())

# Hello! It's nice to meet you. Is there something I can help you with, or would you like to chat?
# 11 25
